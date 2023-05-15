import { Configuration } from "webpack";
import MiniCssExtractPlugin from "mini-css-extract-plugin";
import CopyPlugin from "copy-webpack-plugin";
import path from "path";
export const baseDirectory = "src";

export const config: Configuration = {
  mode: "development",
  entry: {
    main: [`./${baseDirectory}/index.ts`],
  },
  output: {
    path: path.resolve(__dirname, "../dist"),
    filename: "[name].bundle.js",
    publicPath: "/",
  },
  target: "web",
  node: {
    __dirname: false,
    __filename: false,
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: 'public/models/*.onnx', to: '[name][ext]' },
        { from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }
      ],
    }),
  ],
  module: {
    rules: [
      { test: /\.tsx?$/, use: "ts-loader" },
      {
        test: /\.css$/i,
        use: [MiniCssExtractPlugin.loader, "css-loader", "postcss-loader"],
      },
      {
        test: /\.(png|jpe?g|gif)$/i,
        use: [
          {
            loader: "file-loader",
            options: {
              outputPath: "images",
            },
          },
        ],
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
};

export default config;
