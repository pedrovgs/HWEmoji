import { initUIComponents, showCollectDataMode, showTestModelMode, updateEmojiPreview } from "./components/components";
import log from "./log/logger";
import { Gemoji } from "gemoji";
import { Points, WorkMode } from "./domain/model";
import { defaultAppState, selectEmoji, selectMode } from "./domain/state";
import { saveDataSample } from "./data/DataSaver";
import { predictEmoji } from "./ai/HWEmoji";
import Toastify from "toastify-js";

let appState = defaultAppState;

export const init = async () => {
  log("😃 Initializing HWEmoji ");
  const listeners = componentsListener();
  await initUIComponents(listeners, appState);
  log("💪 HWEmoji initialized");
};

const componentsListener = () => {
  const listeners = {
    onEmojiSelected: (gemoji: Gemoji) => {
      appState = selectEmoji(appState, gemoji);
      updateEmojiPreview(gemoji.emoji);
    },
    onModeSelected: (mode: WorkMode) => {
      appState = selectMode(appState, mode);
      if (mode === WorkMode.collectData) {
        showCollectDataMode();
      } else {
        showTestModelMode();
        updateEmojiPreview("🔮");
      }
    },
    onEmojiSaved: (points: Points) => {
      if (points.length == 0) {
        return;
      }
      const emoji = appState.selectedEmoji;
      const sample = {
        emoji: emoji.emoji,
        emojiName: emoji.names[0],
        points: points,
      };
      saveDataSample(sample);
    },
    onPredictionRequested: async (points: Points) => {
      if (points.length == 0) {
        return;
      }
      const predictionResult = await predictEmoji(points);
      const sortedPrediction = predictionResult.sort((a, b) => b.probability - a.probability);
      log(`🔮 Prediction result: ${JSON.stringify(sortedPrediction)}`);
      const firstPredictionProba = sortedPrediction[0].probability;
      if (firstPredictionProba >= 0.9) {
        updateEmojiPreview(sortedPrediction[0].emojiLabel, 300);
      } else if (firstPredictionProba >= 0.7) {
        updateEmojiPreview(sortedPrediction[0].emojiLabel + " or " + sortedPrediction[1].emojiLabel, 100);
      } else {
        updateEmojiPreview("Not sure!", 50);
      }
      const predictionDescription = sortedPrediction.map((p) => `${p.emojiLabel} - ${p.probability}`);
      predictionDescription.forEach((p) => {
        Toastify({
          text: `Prediction probability: ${p}`,
          className: "info",
          duration: 5000,
          close: true,
        }).showToast();
      });
    },
  };
  return listeners;
};
