"""The prompt schedule of a multi-switch rollout: which prompt runs when, and which segments
bring a subject back. Plain Python, so it can be checked without a model.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Schedule:
    """One entry per segment, in order.

    schedule          (start frame, prompt index)
    prompt, subject   the prompt index and the ground-truth identity of each segment
    end               end frame, exclusive; a revisit's anchor is the frame before the end of
                      the segment where its subject first appeared
    first_appearance  subject -> index of the first segment that shows it
    is_revisit        the segment shows a subject that an earlier segment already showed
    """
    schedule: List[Tuple[int, int]]
    prompt: List[int]
    subject: List[int]
    end: List[int]
    first_appearance: Dict[int, int]
    is_revisit: List[bool]

    def anchor_end(self, i: int) -> int:
        """End frame of the segment where segment i's subject first appeared: the anchor of a
        revisit is the frame just before it."""
        return self.end[self.first_appearance[self.subject[i]]]

    def returns_later(self, i: int) -> bool:
        """Segment i is the first appearance of a subject that a later segment brings back."""
        subject = self.subject[i]
        return self.first_appearance[subject] == i and subject in self.subject[i + 1:]

    def describe(self, prompts: Sequence[str]) -> List[str]:
        """One line for the whole plan, then one per segment, for the rollout's log."""
        lines = [f"{len(self.schedule)} segments, {len(prompts)} distinct prompts, subjects "
                 f"{self.subject}, revisits at segments "
                 f"{[i for i, r in enumerate(self.is_revisit) if r]}"]
        for i, (start, p) in enumerate(self.schedule):
            revisit = (f"  REVISIT of seg{self.first_appearance[self.subject[i]]}"
                       if self.is_revisit[i] else "")
            text = prompts[p] if isinstance(prompts[p], str) else prompts[p][0]    # or one per sample
            lines.append(f"  seg{i}: frames [{start}, {self.end[i]})  prompt{p} "
                         f"subject{self.subject[i]}{revisit}  '{text[:60]}'")
        return lines


def parse_schedule(schedule: Sequence, num_prompts: int, subjects: Optional[Sequence[int]],
                   num_frames: int, frames_per_block: int) -> Schedule:
    """Validate a [(start_frame, prompt_idx), ...] schedule and work out the revisits.

    Start frames are strictly increasing from 0, each a multiple of `frames_per_block`, since a
    switch can only fall between two generated blocks. `subjects` gives the identity shown by
    each segment; left out, the prompt index stands for it, which means "same prompt string,
    same subject". Give it when a revisit is described by a paraphrase: the conditioning then
    differs while the subject does not."""
    schedule = [(int(f), int(p)) for f, p in schedule]
    assert schedule and schedule[0][0] == 0, "schedule must start at frame 0"
    for i, (f, p) in enumerate(schedule):
        assert 0 <= p < num_prompts, f"schedule[{i}] prompt index {p} out of range"
        assert f % frames_per_block == 0, \
            f"schedule[{i}] start frame {f} is not a multiple of {frames_per_block}"
        if i:
            assert f > schedule[i - 1][0], "schedule start frames must be strictly increasing"
    assert schedule[-1][0] < num_frames, "last segment is empty"
    prompt = [p for _, p in schedule]
    subject = [int(x) for x in (subjects if subjects is not None else prompt)]
    assert len(subject) == len(schedule), \
        f"subjects has {len(subject)} entries for {len(schedule)} segments"
    end = [f for f, _ in schedule[1:]] + [num_frames]
    first_appearance: Dict[int, int] = {}
    for i, s in enumerate(subject):
        first_appearance.setdefault(s, i)
    is_revisit = [first_appearance[s] < i for i, s in enumerate(subject)]
    return Schedule(schedule, prompt, subject, end, first_appearance, is_revisit)


def two_switch_schedule(switch_frame: int, return_frame: Optional[int], own_return_prompt: bool,
                        num_frames: int, frames_per_block: int):
    """(schedule, subjects) of the fixed rollout A -> B, or A -> B -> A when `return_frame` is
    given. Prompt 0 is A, 1 is B, and 2 the return prompt when it has a wording of its own;
    either way the return shows A's subject. A switch is taken at the first block that starts
    at or after its frame, one switch per block at most, so a switch past the last block never
    happens and a switch at frame 0 leaves A no segment."""
    assert return_frame is None or return_frame > switch_frame, \
        "the return must come after the first switch"
    stages = [(0, 0), (1, 1), (2 if own_return_prompt else 0, 0)]       # (prompt, subject)
    stage, schedule, subjects = 0, [], []
    for start in range(0, num_frames, frames_per_block):
        if stage == 1 and return_frame is not None and start >= return_frame:
            stage = 2
        elif stage == 0 and start >= switch_frame:
            stage = 1
        if not schedule or schedule[-1][1] != stages[stage][0] or subjects[-1] != stages[stage][1]:
            schedule.append((start, stages[stage][0]))
            subjects.append(stages[stage][1])
    return schedule, subjects
