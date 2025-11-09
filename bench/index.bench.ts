import { bench, run, summary } from "mitata";
import { PaddleOcrService } from "../src";

const service = new PaddleOcrService();

await service.initialize();

const imgFile = Bun.file("../assets/receipt.jpg");
const fileBuffer = await imgFile.arrayBuffer();

summary(() => {
  bench("infer test 1", async () => await service.recognize(fileBuffer));
  bench("infer test 2", async () => await service.recognize(fileBuffer));
});

run().then((_) => {
  service.destroy();
});
