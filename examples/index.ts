import { PaddleOcrService } from "../src";
// import { PaddleOcrService } from "ppu-paddle-ocr";

const service = new PaddleOcrService({
  debugging: {
    debug: false,
    verbose: false,
  },
  session: {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
    enableCpuMemArena: true,
    enableMemPattern: true,
    interOpNumThreads: 0,
    intraOpNumThreads: 0,
    executionMode: "sequential",
  },
});
await service.initialize();

const imagePath = "./assets/receipt.jpg";
const imgFile = Bun.file(imagePath);
const fileBuffer = await imgFile.arrayBuffer();

const startTime = Date.now();
const result = await service.recognize(fileBuffer);
const speed = Date.now() - startTime;

service.destroy();

console.log(JSON.stringify(result, null, 2));
console.log(`Operation completed in ${speed} ms`);
