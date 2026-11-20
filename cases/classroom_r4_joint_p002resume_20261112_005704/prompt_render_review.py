"""Validate a recorded human/agent visual review against the exact CVM conditions.

Hashes detect stale reviews; this does not perform semantic image/text review.
"""
from pathlib import Path
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_prompt_render_review(review_path, *, prompt, video, reference):
    review_path = Path(review_path)
    if not review_path.is_file():
        raise ValueError(f'Missing render/prompt visual review: {review_path}')
    review = json.loads(review_path.read_text(encoding='utf-8-sig'))
    if review.get('status') != 'passed' or not review.get('reviewed_by') or not review.get('reviewed_at'):
        raise ValueError('Render/prompt review must record a reviewer, date and passed decision.')
    for name, path in [('prompt', prompt), ('video', video), ('reference', reference)]:
        if review.get('sha256', {}).get(name) != sha(path):
            raise ValueError(f'Stale render/prompt review: {name} changed; visually review the new conditions.')
    if not {0, 31, 62, 92, 123}.issubset(set(review.get('inspected_frames', []))):
        raise ValueError('Review first/intermediate/final frames, including 0/31/62/92/123.')
    required = ('visible_objects_and_returning_landmarks', 'entry_exit_direction_and_final_view',
                'known_content_vs_gray_gaps', 'reference_video_conflicts_resolved_in_prompt',
                'no_inherited_scene_description_conflict')
    if any(review.get('checks', {}).get(k) is not True for k in required):
        raise ValueError('Incomplete render/prompt consistency checklist.')
    if not review.get('observations') or review.get('unresolved_prompt_conflicts') != []:
        raise ValueError('Document visual observations and resolve prompt conflicts before staging.')
    return review
