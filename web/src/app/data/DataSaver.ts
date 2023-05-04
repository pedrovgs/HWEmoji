import { DataSample } from "../domain/model";
import log from "../log/logger";

export const saveDataSample = (sample: DataSample) => {
  log(`Generate raw file with sample data for sample = ${sample.emoji}`);
  const rawData = JSON.stringify(sample);
  const fileUrl = generateTextFileUrl(rawData);
  const linkTag = document.createElement("a");
  linkTag.href = fileUrl;
  const currentTimestamp = Date.now();
  linkTag.download = `${currentTimestamp}-${sample.emojiName}-${sample.emoji}.txt`;
  document.getElementById("body")?.appendChild(linkTag);
  linkTag.click();
  linkTag.remove();
};

const generateTextFileUrl = (txt: string) => {
  let fileData = new Blob([txt], { type: "text/plain" });
  const textFileUrl = window.URL.createObjectURL(fileData);
  return textFileUrl;
};
