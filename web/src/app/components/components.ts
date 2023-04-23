import { Dropdown } from "materialize-css";
import { gemoji, Gemoji } from "gemoji";
import log from "../log/logger";
import { WorkMode, ComponentsListeners, AppState, Point } from "../domain/model";

export const initUIComponents = async (listeners: ComponentsListeners, appState: AppState): Promise<void> => {
  await initMaterializeCssComponents(listeners);
  initializeCanvas();
  updateEmojiPreview(appState.selectedEmoji);
};

export const updateEmojiPreview = (gemoji: Gemoji) => {
  const emojiPreview = document.getElementById("emoji-sample");
  if (emojiPreview != null) {
    emojiPreview.textContent = gemoji.emoji;
  }
};

const initializeCanvas = () => {
  let lastButton = 0;
  let points: Point[][] = [[]];
  const canvas = document.getElementById("whiteboard") as HTMLCanvasElement;
  const canvasBoundingRect = canvas.getBoundingClientRect();
  const canvasLeft = canvasBoundingRect.x;
  const canvasTop = canvasBoundingRect.y;
  if (canvas !== null) {
    const ctx = canvas.getContext("2d");
    canvas.addEventListener("mousemove", (event) => {
      if (event.buttons == 1) {
        const point = { x: event.x - canvasLeft, y: event.y - canvasTop };
        points[points.length - 1].push(point);
        log(`Adding point at position: ${JSON.stringify(point)}`);
        lastButton = 1;
      } else if (event.buttons == 0 && lastButton == 1) {
        points.push([]);
        lastButton = 0;
      }
    });
    if (ctx !== null) {
      // Canvas trick to get better lines
      ctx.translate(0.5, 0.5);
      setInterval(() => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const numberOfPoints = points.length;
        for (var i = 0; i < numberOfPoints; i++) {
          const line = points[i];
          if (line.length <= 0) continue;
          ctx.strokeStyle = "#000000";
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.moveTo(line[0].x, line[0].y);
          for (var k = 1; k < points[i].length; k++) {
            const point = points[i][k];
            ctx.lineTo(point.x, point.y);
          }
          ctx.stroke();
        }
      }, 16);
    }
  }
};

const initMaterializeCssComponents = (listeners: ComponentsListeners): Promise<void> => {
  return new Promise((resolve) => {
    document.addEventListener("DOMContentLoaded", function () {
      var elems = document.querySelectorAll(".dropdown-trigger");
      Dropdown.init(elems);
      initEmojiSelector(listeners);
      initModeSelector(listeners);
      resolve();
    });
  });
};

const initEmojiSelector = (listeners: ComponentsListeners) => {
  const selector = document.getElementById("emoji-selector");
  const selectorTitle = document.getElementById("emoji-selector-title");
  if (selector != null && selectorTitle != null) {
    gemoji.forEach((emojiInfo) => {
      const emojiOption = document.createElement("li");
      const emojiName = emojiInfo.description.charAt(0).toUpperCase() + emojiInfo.description.substring(1);
      emojiOption.innerHTML = `<a>${emojiInfo.emoji}  ${emojiName}</a>`;
      selector.appendChild(emojiOption);
      emojiOption.onclick = () => {
        log(`Emoji selected ${emojiInfo.emoji} - ${emojiName}`);
        selectorTitle.textContent = `Emoji to draw: ${emojiInfo.emoji}`;
        listeners.onEmojiSelected(emojiInfo);
      };
    });
  }
  log(`✅ Emoji selector configured`);
};

const initModeSelector = (listeners: ComponentsListeners) => {
  const modes = [
    { mode: WorkMode.collectData, description: "💪 Collect Data" },
    { mode: WorkMode.test, description: "✅ Test model" },
  ];
  const selector = document.getElementById("mode-selector");
  const selectorTitle = document.getElementById("mode-selector-title");
  if (selector != null && selectorTitle != null) {
    modes.forEach((workMode) => {
      const modeOption = document.createElement("li");
      modeOption.innerHTML = `<a>${workMode.description}</a>`;
      selector.appendChild(modeOption);
      modeOption.onclick = () => {
        log(`Mode selected: ${workMode.description}`);
        selectorTitle.textContent = `Mode option: ${workMode.description}`;
        listeners.onModeSelected(workMode.mode);
      };
    });
  }
  log(`✅ Mode selector configured`);
};
