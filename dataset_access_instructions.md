# Dataset Access Instructions

**Dataset:** Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset

**Source:** Zhang, R., Walshe, C., Liu, Z., Guan, L., Muller, K., Whritner, J., Zhang, L., Hayhoe, M., & Ballard, D. (2020). Atari-HEAD: Atari human eye-tracking and demonstration dataset. *Proceedings of the AAAI Conference on Artificial Intelligence, 34*(04), 6811–6820. https://doi.org/10.1609/aaai.v34i04.6161

**Data DOI:** https://doi.org/10.5281/zenodo.3451402

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Download Instructions

1. Go to https://zenodo.org/records/3451402
2. Download `montezuma_revenge.zip` (431 MB)
3. Place `montezuma_revenge.zip` (do not extract it) into `data/raw/`

The notebook reads frames directly from the zip archive at runtime.

---

## Dataset Contents Used

| File | Contents |
|---|---|
| `montezuma_revenge.zip` | 27 gameplay sessions: 20 standard + 7 highscore |

Each session inside the zip contains:
- A `.tar.bz2` archive of individual gameplay frames as PNG images
- A `.txt` label file with one row per frame: frame ID, timestamp, gaze data, and ALE action code

---

## Session Summary

| Sessions | Count | Frames |
|---|---|---|
| Standard | 20 | ~330,000 |
| Highscore (expert play) | 7 | ~117,000 |
| **Total** | **27** | **447,300** |

---

## Action Mapping

ALE action codes are remapped to 5 classes for this analysis:

| Class | Label | ALE Codes |
|---|---|---|
| 0 | NOOP | 0, 1 |
| 1 | RIGHT | 3, 6, 8, 11, 14, 16 |
| 2 | LEFT | 4, 7, 9, 12, 13, 15 |
| 3 | UP | 2, 10 |
| 4 | DOWN | 5 |

---

## Files Not Used

The full Atari-HEAD dataset contains recordings for 20 Atari games. Only Montezuma's Revenge is used in this analysis.
