import { $$ } from "./selector";

interface Props {
  className: string;
  title: string;
  content: string;
}
export function init(props: Props) {
  const { className, title, content } = props;
  const main = document.querySelector(className);
  const height = window.innerHeight * 0.65;
  const width = height * 0.75;
  const canvas = document.createElement("canvas") as HTMLCanvasElement;
  Object.assign(canvas, { width, height });
  if (main) {
    main.appendChild(canvas);
  }
  const img = new Image();
  img.src = content;
  const context = canvas.getContext("2d");

  if (context) {
    img.onload = function () {
      context.drawImage(img, 0, 0, width, height);
      console.log("drawn");
    };
  }
}
