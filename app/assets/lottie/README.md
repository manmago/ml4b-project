# Exercise animations (Lottie)

The Classify and About tabs show an animation for each exercise. Animations are
**Lottie JSON** files dropped into this folder — no code change is needed. If a
file is missing, the app automatically falls back to a built-in SVG figure, so it
always runs.

## How to add an animation

1. Go to [LottieFiles](https://lottiefiles.com/free-animations/exercise) (or any
   Lottie source) and pick an animation you like — ideally a clean, flat, looping
   one that reads well on a **dark** background.
2. Download it as **Lottie JSON** (the `.json` file, *not* `.lottie`/GIF/MP4).
3. Rename it to the exact class label and place it here.

## Required file names

| File                      | Shown for            |
| ------------------------- | -------------------- |
| `bicep_curl.json`         | Bicep Curl           |
| `tricep_extension.json`   | Tricep Extension     |
| `row.json`                | Row                  |
| `rest.json`               | Rest (optional)      |
| `uncertain.json`          | Uncertain (optional) |
| `unknown.json`            | Unknown (optional)   |

Only `bicep_curl`, `tricep_extension` and `row` matter for the main view; the
rest are optional and fall back to a glyph/figure.

## Notes

- Keep files small (ideally < ~200 KB) so the app stays snappy.
- Record the source + licence of each animation in `docs/DECISIONS.md` when you
  add it (most free LottieFiles animations require attribution).
- These JSON files **are** committed to git — they are app assets, not data.
