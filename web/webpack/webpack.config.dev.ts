import { Configuration, WebpackPluginInstance } from "webpack";
import BrowserSyncPlugin from "browser-sync-webpack-plugin";
import HtmlWebpackPlugin from "html-webpack-plugin";
import MiniCssExtractPlugin from "mini-css-extract-plugin";
import { Configuration as DevServerConfiguration } from "webpack-dev-server";
import { config } from "./webpack.config.common";
import path from "path";

const dev: Configuration & { devServer: DevServerConfiguration } = {
  ...config,
  devtool: "inline-source-map",
  devServer: {
    compress: true,
    hot: true,
    port: 9001,
    static: false,
    watchFiles: ["src/**/*"],
  },
  plugins: (config.plugins ?? []).concat([
    new HtmlWebpackPlugin({
      title: "Home",
      hash: true,
      filename: "index.html",
      chunks: ["main"],
      publicPath: "auto",
      template: path.join(__dirname, "../public/index.html"),
    }),
    new MiniCssExtractPlugin({
      filename: "[name].css",
      chunkFilename: "style.css",
    }),
    new BrowserSyncPlugin(
      {
        port: 3000,
        proxy: "http://localhost:9001",
        open: false,
      },
      { reload: true },
    ) as unknown as WebpackPluginInstance,
  ]),
};

export default dev;
