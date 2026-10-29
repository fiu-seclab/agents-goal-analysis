const express = require('express')
const bodyParser = require("body-parser");
const cors = require('cors');
const uuid = require('uuid');
const mongoose = require('mongoose');
const mongoConfig = require('./config/mongo');

const eventsUtil = require('./utils/events');

const app = require('express')();

const server = require('http').Server(app);
const io = require('socket.io')(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST", "PUT"],
    allowedHeaders: ["custom-header"],
    credentials: true
  }
});

async function main() {
  io.on('connection', socket => {
    const cookie = {
      id: uuid.v4(),
    };
    // console.log('Generate and Send Cookie', cookie.id);
    socket.emit('auth', cookie);

    socket.on('data', data => {
      if (data.data){
        data = data.data
      }
      eventsUtil.saveEventsInMongo(socket.handshake.headers, data);
    });
    
  });

  app.use(bodyParser.urlencoded({ extended: false }));
  app.use(bodyParser.json());
  app.use(cors({
    origin: '*',
    //credentials: true,
  }));

  app.use(express.static('public'));

  server.listen(8080, () => {
    console.log('App listening at http://localhost:8080');
  });

  // await mongoose.connect(`mongodb://behx:behx1234@${mongoConfig.url}/${mongoConfig.db}?authSource=admin`)
  await mongoose.connect(`mongodb://${mongoConfig.url}/${mongoConfig.db}`)
}


main().catch(err => {
  console.log(err);
})
