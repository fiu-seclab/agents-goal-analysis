const OUTPUT_FORMAT = process.env.OUTPUT_FORMAT;
const fs = require('fs');
const os = require('os');
const Event = require('../models/event');

const methods = {
    saveEventsInMongo: async (headers, data) => {
        const userId = data.uid;
        const url = data.url;
        const info = unescape(data.info);
        const infoChucks = info.split('|||')
        infoChucks.forEach(item => {
            const args = item.split(' ');
            const temp = item.split('}')[0];
            const attrs = temp.substring(temp.indexOf('{') + 1);
            const values = {
                userId,
                eventName: args[4],
                timestamp: args[1],
                pos: {
                    x: Number(args[2]),
                    y: Number(args[3]),
                },
                cursor: Number(args[0]),
                element: args[5],
                attrs,
                extra: args[args.length - 1],
                fingerprint: {
                    source: headers['x-real-ip'],
                    userAgent: headers['user-agent'],
                }
            };
            
            values.url = headers.origin;
            
            if (data.url) {
                values.url = data.url;
            }
            // console.log(values.eventName, values.pos.x, values.pos.y);
            // console.log(values.userId);
            const event = new Event(values);
            event.save();
        });
    },
    saveEventsInFile: async (hostname, data) => {
        const info = unescape(data.data.info);
        const infoChucks = info.split('|||')
        if (OUTPUT_FORMAT == 'FILE') {
        for (chunk of infoChucks) {
            fs.appendFile(`prod_logs/${hostname}-${data.data.uid}.log`, chunk + os.EOL, err => {
            if (err) {
                console.error(err)
                return
            }
            })
        }
        } else {
            console.log(data.data.uid);
            console.log(infoChucks);
            console.log('-------------------------');
        }
    },
}


module.exports = methods;
