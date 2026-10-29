const mongoose = require('mongoose');

const eventSchema = new mongoose.Schema({
    userId: String,
    eventName: String,
    timestamp: String,
    pos: {
        x: Number,
        y: Number,
    },
    cursor: Number,
    element: String,
    attrs: String,
    extra: String,
    url: String,
    fingerprint: {
        source: String,
        userAgent: String,
    }
});

const Event = mongoose.model('Event', eventSchema);

module.exports = Event;