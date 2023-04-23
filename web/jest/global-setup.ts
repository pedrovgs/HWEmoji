module.exports = async () => {
  console.log("HTML page testing...");
  console.log(`environment variables: ${process.env.NODE_ENV}`);
  process.env.TZ = "Asia/Seoul";
};
