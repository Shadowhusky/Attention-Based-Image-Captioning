const path = require('path');
const express = require("express");
const app = express();
const port = 3000;
app.use(express.static("src"))

app.listen(port, () => {
  console.log(`Listening at http://localhost:${port}`);
});

function showImage(img, caption) {
  app.get("/img", function (req, res) {
    res.set({ "Content-Type": "image/jpg" });
    res.sendFile(__dirname + img.substring(1));
  });
  app.get("/caption", function (req, res) {
    res.send(caption);
  });
}

module.exports = showImage;
