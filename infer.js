/* 浏览器端大豆病害识别：onnxruntime-web + MobileNetV3 ONNX。
 * 输入图片 -> canvas 缩放 224x224 -> ImageNet 归一化 -> NCHW -> 推理 -> softmax。
 */
const ORT_WASM_DIR = "ort/";
let ortSession = null;
let KB = null;

async function initModel() {
  if (ortSession) return ortSession;
  ort.env.wasm.wasmPaths = ORT_WASM_DIR;
  // GitHub Pages 无 COOP/COEP 头，SharedArrayBuffer 不可用，强制单线程避免回退警告
  ort.env.wasm.numThreads = 1;
  ortSession = await ort.InferenceSession.create("model.onnx", {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return ortSession;
}

async function loadKnowledge() {
  if (KB) return KB;
  const r = await fetch("knowledge.json");
  KB = await r.json();
  return KB;
}

// 图片 -> Float32Array(NCHW, normalized)
function preprocess(imgEl) {
  const size = 224;
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(imgEl, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;
  const mean = KB.normalize.mean;
  const std = KB.normalize.std;
  const out = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    const r = data[i * 4] / 255, g = data[i * 4 + 1] / 255, b = data[i * 4 + 2] / 255;
    out[i] = (r - mean[0]) / std[0];
    out[size * size + i] = (g - mean[1]) / std[1];
    out[2 * size * size + i] = (b - mean[2]) / std[2];
  }
  return out;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

async function predict(imgEl) {
  await Promise.all([initModel(), loadKnowledge()]);
  const input = preprocess(imgEl);
  const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);
  const results = await ortSession.run({ input: tensor });
  const logits = Array.from(results.logits.data);
  const probs = softmax(logits);
  const classes = KB.classes;
  const ranked = classes.map((cls, i) => ({
    cls,
    class: cls,
    confidence: Math.round(probs[i] * 10000) / 100,
  })).sort((a, b) => b.confidence - a.confidence);
  const top = ranked[0];
  const info = KB.info[top.cls] || {};
  return {
    mode: "model",
    top: { class: top.cls, name_zh: info.name_zh || top.cls, confidence: top.confidence },
    candidates: ranked.slice(0, 4).map(x => ({
      class: x.cls,
      name_zh: (KB.info[x.cls] || {}).name_zh || x.cls,
      confidence: x.confidence,
    })),
    detail: {
      name_zh: info.name_zh || top.cls,
      name_en: info.name_en || top.cls,
      pathogen: info.pathogen || "",
      type: info.type || "",
      symptoms: info.symptoms || "",
      confusion: info.confusion || "",
      control: info.control || [],
      color: info.color || "#2e9e5b",
      emoji: info.emoji || "🌿",
    },
  };
}
