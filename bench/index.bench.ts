import { bench, run, summary } from "mitata";
import { PaddleOcrService } from "../src";

const service = new PaddleOcrService();

await service.initialize();

const imgFile = Bun.file("./assets/receipt.jpg");
const deskewFile = Bun.file("./assets/tilted.png");

const fileBuffer = await imgFile.arrayBuffer();
const deskewBuffer = await deskewFile.arrayBuffer();

summary(() => {
  bench("infer test 1", async () => await service.recognize(fileBuffer));
  bench("infer test 2", async () => await service.recognize(fileBuffer));
});

summary(() => {
  bench(
    "infer deskew test 1",
    async () => await service.deskewImage(deskewBuffer)
  );
  bench(
    "infer deskew test 2",
    async () => await service.deskewImage(deskewBuffer)
  );
});

run().then((_) => {
  service.destroy();
});
