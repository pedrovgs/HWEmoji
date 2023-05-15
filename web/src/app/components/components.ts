import { Dropdown } from "materialize-css";
import { gemoji, Gemoji } from "gemoji";
import log from "../log/logger";
import { WorkMode, ComponentsListeners, AppState, Point } from "../domain/model";
import { emojisSupported } from "../ai/emojis-supported";

export const initUIComponents = async (listeners: ComponentsListeners, appState: AppState): Promise<void> => {
  await initMaterializeCssComponents(listeners);
  const canvas = initializeCanvas();
  initializeEmojisSupportedList();
  initializeSaveButton(listeners, canvas);
  initializePredictButton(listeners);
  initializeCleanButton(canvas);
  updateEmojiPreview(appState.selectedEmoji.emoji);
};

export const updateEmojiPreview = (emoji: string) => {
  const emojiPreview = document.getElementById("emoji-sample");
  if (emojiPreview != null) {
    emojiPreview.textContent = emoji;
  }
};

export const showCollectDataMode = () => {
  showSaveButton(true);
  showEmojiSelector(true);
  showPredictButton(false);
  showEmojisSupportedDescription(false);
};

export const showTestModelMode = () => {
  showSaveButton(false);
  showEmojiSelector(false);
  showPredictButton(true);
  showEmojisSupportedDescription(true);
};

const initializeEmojisSupportedList = () => {
  const emojis = emojisSupported.join("   ");
  const description = document.getElementById("emojis-supported-list");
  if (description) {
    description.textContent = "Emojis supported: " + emojis;
    description.style.display = "none";
  }
};

const showEmojisSupportedDescription = (show: boolean) => {
  const saveButton = document.getElementById("emojis-supported-list");
  if (saveButton) {
    saveButton.style.display = show ? "" : "none";
  }
};

const showSaveButton = (show: Boolean) => {
  const saveButton = document.getElementById("save-sample-button");
  if (saveButton) {
    saveButton.style.display = show ? "" : "none";
  }
};

const showPredictButton = (show: Boolean) => {
  const button = document.getElementById("predict-button");
  if (button) {
    button.style.display = show ? "" : "none";
  }
};

const showEmojiSelector = (show: Boolean) => {
  const selector = document.getElementsByClassName("emoji-selector")[0] as HTMLElement;
  if (selector) {
    selector.hidden = !show;
  }
};

let points: Point[][] = [[]];
let modifiedSinceLastDraw = false;

const initializeCanvas = (): HTMLCanvasElement => {
  let lastButton = 0;
  const canvas = document.getElementById("whiteboard") as HTMLCanvasElement;
  const canvasBoundingRect = canvas.getBoundingClientRect();
  const canvasLeft = canvasBoundingRect.x;
  const canvasTop = canvasBoundingRect.y;
  if (canvas !== null) {
    const ctx = canvas.getContext("2d");
    canvas.addEventListener("pointermove", (event) => {
      if (event.buttons == 1) {
        const coalescedEvents = event.getCoalescedEvents();
        coalescedEvents.forEach((coalescedEvent) => {
          const coalescedPoint = { x: coalescedEvent.x - canvasLeft, y: coalescedEvent.y - canvasTop };
          points[points.length - 1].push(coalescedPoint);
        });
        lastButton = 1;
        modifiedSinceLastDraw = true;
      } else if (event.buttons == 0 && lastButton == 1) {
        points.push([]);
        lastButton = 0;
      }
    });
    if (ctx !== null) {
      setInterval(() => {
        if (!modifiedSinceLastDraw) return;
        resetCanvasContent(canvas);
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
        modifiedSinceLastDraw = false;
      }, 16);
    }
  }
  return canvas;
};

const initializeSaveButton = (listeners: ComponentsListeners, canvas: HTMLCanvasElement) => {
  document.getElementById("save-sample-button")?.addEventListener("click", () => {
    const plainPoints = points.flat();
    listeners.onEmojiSaved(plainPoints);
    resetSavedPoints();
    resetCanvasContent(canvas);
  });
};

const initializePredictButton = (listeners: ComponentsListeners) => {
  const button = document.getElementById("predict-button");
  if (button) {
    button.addEventListener("click", () => {
      const plainPoints = points.flat();
      listeners.onPredictionRequested(plainPoints);
    });
    button.style.display = "none";
  }
};

const initializeCleanButton = (canvas: HTMLCanvasElement) => {
  document.getElementById("clear-whiteboard-button")?.addEventListener("click", () => {
    resetSavedPoints();
    resetCanvasContent(canvas);
  });
};

const resetSavedPoints = () => {
  points = [[]];
};

const resetCanvasContent = (canvas: HTMLCanvasElement) => {
  const ctx = canvas.getContext("2d");
  ctx?.clearRect(0, 0, canvas.width, canvas.height);
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
        selectorTitle.textContent = `Mode: ${workMode.description}`;
        listeners.onModeSelected(workMode.mode);
      };
    });
  }
  log(`✅ Mode selector configured`);
};
