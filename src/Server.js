const express = require('express')
const app = express()
const port = 3000;

app.use(express.static(__dirname+'/'));

function showImage(img) {
  app.get('/img', function (req, res) {
    res.set({'Content-Type': 'image/jpg'})
    console.log(__dirname + img.substring(1))
    res.sendFile(__dirname + img.substring(1));
  })
  app.listen(port, () => console.log('The server running on Port '+port));
}

exports.showImage = showImage;