# Local restricted overlays

Tracked SparkDuet stays on published licenses: MIT, Apache, ShapleyMcg,
Qwen Community, and the grants named in each lane file. That is the
recipe a clone of this repo boots.

Some cards are not a clean public grant (CC BY-NC-ND, research-only,
unpublished commercial terms, `license: other` with no MIT/Apache/BSD
text). Those stay off GitHub. This page is the public hook so a
reinstall still knows where a local override lives.

## How the hook works

`local/restricted/` is gitignored. `sparkduetctl` (`compose_rel` in
`scripts/sparkduetctl.sh`) prefers a same-named file there over
`configs/`. Remove the local file and the tracked compose is back.

`scripts/test_lane_glm.py` asserts three things: public Lane G EXL3 is
MTP, that tree is gitignored, and ctl has the hook. CI fails if a
restricted compose lands in `configs/`.

Do not advertise a restricted speculator as a house lane in README,
MODELS.md, Command Center, or the house catalog. The served model id
can stay the same; the drafter is what must not be sold as the
commercial recipe.

## Recreate the public Lane G EXL3 hop

This is what GitHub ships.

1. `cp configs/sparkduet.env.example sparkduet.env` and set the pair,
   fabric, and `G_ENGINE=exl3`.
2. `./scripts/prepare-models.sh --model glm-exl3` on the head (pins
   `brandonmusic/GLM-5.3-Flash-tr3-4bpw` rev `5ab363a8`, syncs to the
   worker).
3. Pull `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` on both
   boxes.
4. `./scripts/sparkduetctl.sh doctor` then `switch glm`. Pair only.
   Never start a lone GLM head.
5. Prove `GET :30000/v1/models` and a short completion. Public compose
   is MTP-2. Recipe door is 262144 at util 0.80.

Details: `docs/MODELS.md` (Lane G EXL3), `configs/lane-glm-exl3.compose.yml`,
`patches/glm-exl3-sm121/README.md`.

## Recreate a research overlay

The overlay itself is not in this repository. Keep a copy of
`local/restricted/` off-git: this pair stores it on the Mac clone and
at `/srv/ai/sparkduet/local/restricted/` on both Sparks. That directory
is the recipe (compose, entry, extra patches, fetch script, measured
numbers). Copy it back onto a rebuilt box, confirm `compose_rel` is in
ctl, then `switch glm`.

If the local compose is missing on either node, ctl silently uses the
public file. Both boxes must match.

## Turn it off

Delete or rename the local compose on both boxes, `switch glm`, and
re-apply house pickers so they clip to the live `/v1/models`
`max_model_len`.
