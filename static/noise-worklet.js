// AudioWorklet: real-time noise gate + downward expander for the live preview.
//
// Web Audio has no built-in noise gate or denoiser, so the "Noise gate" and
// "Noise reduction" sliders had nothing to drive client-side. This processor
// gives them an audible, real effect in the live preview:
//
//   - gateThreshold (dB): below this envelope level the signal is silenced
//     (a hard noise gate) -- cuts hiss/rumble between words.
//   - reduction (0..1): strength of a downward expander that pulls down the
//     low-level noise floor sitting under/around the speech. 0 = transparent,
//     1 = aggressive.
//
// This is an approximation of the server-side spectral denoise (DeepFilterNet),
// not the same algorithm -- it works in the time domain -- but it makes the
// sliders genuinely change what you hear while auditioning.

class NoiseReducerProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: "gateThreshold", defaultValue: -60, automationRate: "k-rate" }, // dB
      { name: "reduction", defaultValue: 0.0, automationRate: "k-rate" },      // 0..1
    ];
  }

  constructor() {
    super();
    this.env = 0;      // envelope follower state
    this.gain = 1;     // smoothed gain state
  }

  process(inputs, outputs, params) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || input.length === 0) return true;

    const gateTh = Math.pow(10, params.gateThreshold[0] / 20); // linear amplitude
    const reduction = Math.min(1, Math.max(0, params.reduction[0]));
    // Expander threshold rises with reduction: -50 dB (gentle) .. -18 dB (strong).
    const expTh = Math.pow(10, (-50 + reduction * 32) / 20);
    const ratio = 1 + reduction * 5; // up to 6:1 downward expansion

    const atk = 0.02;   // envelope attack coefficient
    const rel = 0.0025; // envelope release coefficient
    const gAtk = 0.3;   // gain smoothing when opening
    const gRel = 0.05;  // gain smoothing when closing

    const nchan = input.length;
    // Use channel 0 to drive a shared envelope so all channels gate together.
    const drive = input[0];
    const frames = drive.length;
    let env = this.env;
    let gain = this.gain;

    for (let i = 0; i < frames; i++) {
      const a = Math.abs(drive[i]);
      env += (a > env ? atk : rel) * (a - env);

      let target = 1;
      if (env < gateTh) {
        target = 0;
      } else if (env < expTh && reduction > 0) {
        // downward expansion: reduce gain the further below expTh we are
        const belowDb = 20 * Math.log10(expTh / Math.max(env, 1e-8));
        const reduceDb = belowDb * (ratio - 1) / ratio;
        target = Math.pow(10, -reduceDb / 20);
      }

      gain += (target > gain ? gAtk : gRel) * (target - gain);

      for (let ch = 0; ch < nchan; ch++) {
        output[ch][i] = input[ch][i] * gain;
      }
    }

    this.env = env;
    this.gain = gain;
    return true;
  }
}

registerProcessor("noise-reducer", NoiseReducerProcessor);
