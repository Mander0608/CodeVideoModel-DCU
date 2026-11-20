"""Official generation with DCU attention and bounded-memory offload."""
import faulthandler, hashlib, itertools, json, os, signal, subprocess, sys, time
from pathlib import Path
import torch
from cvm.dcu_runtime import configure
configure()
faulthandler.register(signal.SIGUSR1, all_threads=True)
from cvm import video

NativeH3 = video.H3
class DcuH3(NativeH3):
    def load(self, model_root):
        start=time.monotonic()
        super().load(model_root)
        if os.environ.get('CVM_SMOKE_TEST') == '1':
            self.generation.register_to_config(reference_image_short_edge=512)
        # ParaStor-backed mmap -> HIP transfers are much slower than transfers
        # from anonymous/pinned RAM. Materialize once, preserving every value.
        for name, module in [('text_encoder',self.conditioner.text_encoder),
                             ('vae',self.generation.vae),('audio_vae',self.generation.audio_vae),
                             ('transformer',self.transformer)]:
            copied=0; last=0; begin=time.monotonic()
            print('PIN_START',name,flush=True)
            for tensor in itertools.chain(module.parameters(),module.buffers()):
                if tensor.device.type=='cpu' and not tensor.is_pinned():
                    tensor.data=tensor.detach().pin_memory()
                    copied+=tensor.numel()*tensor.element_size()
                    if copied-last>5e9:
                        print('PIN_PROGRESS',name,round(copied/1e9,2),'GB',flush=True)
                        last=copied
            print('PIN_DONE',name,round(copied/1e9,2),'GB',round(time.monotonic()-begin,3),'sec',flush=True)
        # Each ~62 GiB model leaves insufficient activation space on a 64 GiB device.
        # Diffusers supports group offload together with ComponentsManager hooks.
        self.transformer.enable_group_offload(
            onload_device=self.device, offload_device=torch.device('cpu'),
            offload_type='block_level', num_blocks_per_group=2, use_stream=False)
        # Text encoding is run only twice; leaf offload trades speed for bounded memory.
        self.managers[0].disable_auto_cpu_offload()
        from accelerate import cpu_offload
        cpu_offload(self.conditioner.text_encoder, execution_device=self.text_device, offload_buffers=True)
        for index, layer in enumerate(self.conditioner.text_encoder.model.language_model.layers):
            if index == 0 or (index+1) % 8 == 0:
                def completed(module, inputs, output, index=index):
                    print('TEXT_LAYER_DONE',index+1,'epoch',round(time.time(),3),flush=True)
                layer.register_forward_hook(completed)
        self.step_index=0
        print('MODEL_READY', round(time.monotonic()-start,3), flush=True)

    def proxy(self,*args,**kwargs):
        print('ENCODE_PROXY_START',flush=True)
        result=super().proxy(*args,**kwargs)
        if not torch.isfinite(result[1]).all():
            raise FloatingPointError('Proxy VAE returned non-finite latents')
        print('ENCODE_PROXY_DONE',list(result[1].shape),flush=True)
        return result

    @torch.inference_mode()
    def close(self):
        # Accelerate leaves offloaded parameters on meta. Restore its CPU
        # weights before ComponentsManager.remove() calls module.to('cpu').
        conditioner = getattr(self, 'conditioner', None)
        if conditioner is not None:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(conditioner.text_encoder, recurse=True)
        super().close()

    def image(self,*args,**kwargs):
        print('ENCODE_IMAGE_START',flush=True)
        result=super().image(*args,**kwargs)
        if not torch.isfinite(result[1]).all():
            raise FloatingPointError('Image VAE returned non-finite latents')
        print('ENCODE_IMAGE_DONE',list(result[1].shape),flush=True)
        return result

    def condition(self,*args,**kwargs):
        print('ENCODE_CONDITION_START',flush=True)
        result=super().condition(*args,**kwargs)
        if not torch.isfinite(result.embeds).all():
            raise FloatingPointError('Text encoder returned non-finite embeddings')
        print('ENCODE_CONDITION_DONE',list(result.embeds.shape),flush=True)
        return result

    def decode(self,*args,**kwargs):
        print('DECODE_START',flush=True)
        result=super().decode(*args,**kwargs)
        print('DECODE_DONE',flush=True)
        return result

    def predict(self,*args,**kwargs):
        start=time.monotonic()
        result=super().predict(*args,**kwargs)
        if not all(torch.isfinite(tensor).all().item() for tensor in result):
            raise FloatingPointError(f'Non-finite prediction at step {self.step_index+1}')
        torch.cuda.synchronize(self.device)
        self.step_index+=1
        print('DENOISE',self.step_index,'seconds',round(time.monotonic()-start,3),
              'allocated_GiB',round(torch.cuda.memory_allocated(self.device)/2**30,3),flush=True)
        return result

video.H3=DcuH3
if __name__=='__main__':
    from inference.generate_video import main
    start=time.monotonic()
    main()
    elapsed=time.monotonic()-start
    output=Path(sys.argv[sys.argv.index('--output')+1])
    media=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(output)]))
    summary={'status':'completed','smoke_test':os.environ.get('CVM_SMOKE_TEST')=='1','elapsed_seconds':elapsed,'torch':torch.__version__,
             'hip':torch.version.hip,'argv':sys.argv,'media':media,
             'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
             'peak_allocated_GiB':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())]}
    output.with_suffix('.run.json').write_text(json.dumps(summary,indent=2))
    print('RUN_COMPLETE_SECONDS',round(elapsed,3),flush=True)
