# GPS L1 jammer classifier: model card


## What it does
Ten times per second the app takes 4096 samples (1.7 ms) of the 2.4 MHz around GPS L1
(1575.42 MHz) and turns them into a 64 x 64 time-frequency picture: 64 consecutive 64-point FFTs
(Hann window, 37.5 kHz bins, 26.7 us frames), in dB above the learned quiet-band reference.
A small convolutional neural network decides which of 5 classes the picture shows. It runs on the
CPU inside the app (ONNX Runtime) and needs no internet. The app averages the probabilities over
one second before it raises an event.

## Data
- **Generator:** a C# dataset generator uses the same `L1Spectrogram` code as the live app.
  Each capture gets receiver noise at a random gain (3-14 ADC counts RMS), one interferer, 8-bit
  quantisation with clipping like the RTL2832U ADC, and a quiet-band reference measured from a
  separate noise-only capture with a random +-0.5 dB calibration error.
- **Training set:** seed 1, 10000 captures (8500 train / 1500 validation for early stopping).
- **Test set:** seed 2, 3330 captures, never used for training.
- **Classes and generator ranges** (JNR = jammer-to-noise ratio in the 2.4 MHz receiver bandwidth, 0-30 dB):
  - Clean: receiver noise only; 40% of captures include a weak internal spur (-25 to -10 dB).
  - CW tone: one tone anywhere in +-1.05 MHz, slow drift, a little phase noise.
  - Swept chirp: sawtooth or triangle sweeps 1.5-30 MHz wide, 8-400 us period (wider than the
    dongle's band, faded out by the tuner's passband like real hardware).
  - Wideband noise: Gaussian noise 1-20 MHz wide.
  - Pulsed: tone or noise bursts, 100 us-1.2 ms repetition interval, 5-50% duty cycle.
- **Augmentation during training:** random circular frequency shift and small time shift.

## Model
Input `[N, 1, 64, 64]` -> 3 x (3x3 convolution, ReLU, 2x2 max-pool: 8, 16, 32 channels) ->
dense 2048 -> 32 -> 5, softmax; 71621 parameters. Cross-entropy, Adam (lr 2e-3, weight
decay 1e-4), batch 128, dropout 0.2, early stopping on validation loss
(60 epochs, 1450 s on a CPU). Inference: 8.29 ms per capture (PyTorch, CPU, batch 1).

## Results on the held-out test set
| | Accuracy | Macro F1 | Jammer detection rate | False alarms on clean band |
|---|---|---|---|---|
| **CNN** | **0.992** | **0.992** | **0.996** | **0.005** |
| Rule-based baseline | 0.730 | 0.699 | 0.981 | 0.000 |

The baseline (`RuleJammerClassifier`) uses five hand-measured features: band power rise, peak
excess, fraction of hot bins, frame-power variation over time and spectral flatness.

### Accuracy by jammer strength (jammer classes only)
| JNR | CNN | Rule baseline | Captures |
|---|---|---|---|
| 0-5 dB | 0.961 | 0.552 | 464 |
| 5-10 dB | 0.995 | 0.705 | 438 |
| 10-20 dB | 0.998 | 0.679 | 876 |
| 20-30 dB | 0.997 | 0.684 | 886 |

### Per class (CNN)
| Class | Precision | Recall | F1 | Test captures |
|---|---|---|---|---|
| Clean | 0.985 | 0.995 | 0.990 | 666 |
| CW tone | 1.000 | 1.000 | 1.000 | 666 |
| Swept chirp | 0.991 | 0.968 | 0.979 | 666 |
| Wideband noise | 0.996 | 1.000 | 0.998 | 666 |
| Pulsed | 0.987 | 0.994 | 0.990 | 666 |

### Confusion matrix (CNN)
| true \ predicted | Clean | CW tone | Swept chirp | Wideband noise | Pulsed |
|---|---|---|---|---|---|
| **Clean** | 663 | 0 | 3 | 0 | 0 |
| **CW tone** | 0 | 666 | 0 | 0 | 0 |
| **Swept chirp** | 9 | 0 | 645 | 3 | 9 |
| **Wideband noise** | 0 | 0 | 0 | 666 | 0 |
| **Pulsed** | 1 | 0 | 3 | 0 | 662 |

### Confusion matrix (rule baseline)
| true \ predicted | Clean | CW tone | Swept chirp | Wideband noise | Pulsed |
|---|---|---|---|---|---|
| **Clean** | 666 | 0 | 0 | 0 | 0 |
| **CW tone** | 0 | 666 | 0 | 0 | 0 |
| **Swept chirp** | 32 | 0 | 80 | 205 | 349 |
| **Wideband noise** | 0 | 0 | 246 | 420 | 0 |
| **Pulsed** | 18 | 8 | 12 | 28 | 600 |

## Why the CNN wins
A wide, slow chirp sweeps through the 2.4 MHz window only part of the time, so in the power-over-time
features it looks like a pulsed jammer. The CNN sees the diagonal line inside each "pulse" and
tells them apart. A fast chirp (period shorter than an FFT frame) appears as a comb of lines, which
the CNN also separates from flat wideband noise.
