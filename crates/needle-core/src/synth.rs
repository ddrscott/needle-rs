//! Templated training data for the bundled environments.
//!
//! The reference fine-tunes on data an LLM writes from the tool schemas
//! (`needle generate-data`). This is the offline equivalent for the
//! `smart_home` surface: paraphrase templates over every call shape and
//! every refusal category the acceptance suite checks (missing value,
//! irrelevant, negated, out of range), plus two-call requests. Queries that
//! appear verbatim in the frozen suite are never emitted.

use serde_json::{Map, Value, json};

use crate::render::py_strip;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
}

const ROOMS: [(&str, &[&str]); 4] = [
    ("kitchen", &["kitchen"]),
    ("living_room", &["living room", "lounge", "family room"]),
    ("bedroom", &["bedroom", "master bedroom"]),
    ("study", &["study", "den"]),
];
const FAN_ROOMS: [&str; 3] = ["living_room", "bedroom", "study"];
const VAC_ROOMS: [&str; 3] = ["kitchen", "living_room", "bedroom"];
const COLORS: [&str; 5] = ["warm white", "cool white", "red", "green", "blue"];
const SPEEDS: [&str; 3] = ["low", "medium", "high"];

fn room_word(rng: &mut Rng, room: &str) -> String {
    let words = ROOMS.iter().find(|(r, _)| *r == room).map(|(_, w)| *w).unwrap_or(&[]);
    rng.pick(words).to_string()
}

fn call(name: &str, args: Value) -> Value {
    json!({"name": name, "arguments": args})
}

/// One positive request: `(query, calls, reasoning)`.
fn positive(rng: &mut Rng) -> (String, Vec<Value>, String) {
    match rng.below(9) {
        0 | 1 => {
            let (room, _) = *rng.pick(&ROOMS);
            let w = room_word(rng, room);
            let on = rng.chance(0.5);
            let a = if on { "on" } else { "off" };
            let q = match rng.below(6) {
                0 => format!("turn {a} the {w} lights"),
                1 => format!("switch {a} the lights in the {w}"),
                2 => format!("lights {a} in the {w} please"),
                3 => format!("can you turn the {w} lights {a}"),
                4 => format!("{} the {w} lights", if on { "power up" } else { "kill" }),
                _ => format!("please switch the {w} lights {a}"),
            };
            let r = format!("room '{w}' -> {room}; action {a} from query.");
            (q, vec![call("control_lights", json!({"room": room, "action": a}))], r)
        }
        2 => {
            let (room, _) = *rng.pick(&ROOMS);
            let w = room_word(rng, room);
            let pct = 5 * (1 + rng.below(19));
            let q = match rng.below(5) {
                0 => format!("dim the {w} lights to {pct} percent"),
                1 => format!("set the {w} lights to {pct}%"),
                2 => format!("bring the {w} lights down to {pct} percent"),
                3 => format!("dim the lights in the {w} to {pct}%"),
                _ => format!("make the {w} lights {pct} percent bright"),
            };
            let r = format!("room '{w}' -> {room}; dim to {pct} from '{pct}'.");
            (q, vec![call("control_lights", json!({"room": room, "action": "dim", "brightness_percent": pct}))], r)
        }
        3 => {
            let (room, _) = *rng.pick(&ROOMS);
            let w = room_word(rng, room);
            let color = *rng.pick(&COLORS);
            let q = match rng.below(4) {
                0 => format!("turn the {w} lights {color}"),
                1 => format!("make the {w} lights {color}"),
                2 => format!("switch on the {w} lights in {color}"),
                _ => format!("set the {w} lights to {color}"),
            };
            let r = format!("room '{w}' -> {room}; color '{color}' means action on.");
            (q, vec![call("control_lights", json!({"room": room, "action": "on", "color": color}))], r)
        }
        4 => {
            let t = 10 + rng.below(21);
            let q = match rng.below(6) {
                0 => format!("set the thermostat to {t} degrees"),
                1 => format!("make it {t} degrees in here"),
                2 => format!("change the heating to {t}"),
                3 => format!("thermostat {t} please"),
                4 => format!("heat the house to {t} degrees"),
                _ => format!("I want the temperature at {t} degrees"),
            };
            let r = format!("temperature {t} from '{t}'.");
            (q, vec![call("set_thermostat", json!({"temperature": t}))], r)
        }
        5 => {
            let room = *rng.pick(&FAN_ROOMS);
            let w = room_word(rng, room);
            let on = rng.chance(0.6);
            let a = if on { "on" } else { "off" };
            if on && rng.chance(0.5) {
                let sp = *rng.pick(&SPEEDS);
                let q = match rng.below(3) {
                    0 => format!("turn on the {w} fan on {sp}"),
                    1 => format!("set the {w} fan to {sp} speed"),
                    _ => format!("run the fan in the {w} at {sp}"),
                };
                let r = format!("room '{w}' -> {room}; fan on at {sp}.");
                (q, vec![call("control_fan", json!({"room": room, "action": "on", "speed": sp}))], r)
            } else {
                let q = match rng.below(4) {
                    0 => format!("turn {a} the {w} fan"),
                    1 => format!("switch the fan in the {w} {a}"),
                    2 => format!("{w} fan {a}"),
                    _ => format!("please turn the {w} fan {a}"),
                };
                let r = format!("room '{w}' -> {room}; fan {a}.");
                (q, vec![call("control_fan", json!({"room": room, "action": a}))], r)
            }
        }
        6 => {
            let (room, _) = *rng.pick(&ROOMS);
            let w = room_word(rng, room);
            let open = rng.chance(0.5);
            let a = if open { "open" } else { "close" };
            let q = match rng.below(5) {
                0 => format!("{a} the {w} blinds"),
                1 => format!("{a} the blinds in the {w}"),
                2 => format!("please {a} the {w} blinds"),
                3 => format!("{} the {w} blinds", if open { "raise" } else { "lower" }),
                _ => format!("could you {a} the blinds in the {w}"),
            };
            let r = format!("room '{w}' -> {room}; blinds {a}.");
            (q, vec![call("control_blinds", json!({"room": room, "action": a}))], r)
        }
        _ => {
            let kind = rng.below(4);
            if kind == 0 {
                let room = *rng.pick(&VAC_ROOMS);
                let w = room_word(rng, room);
                let q = match rng.below(3) {
                    0 => format!("vacuum the {w}"),
                    1 => format!("start the vacuum in the {w}"),
                    _ => format!("run the robot vacuum in the {w}"),
                };
                let r = format!("start the vacuum; room '{w}' -> {room}.");
                (q, vec![call("start_robot_vacuum", json!({"action": "start", "room": room}))], r)
            } else {
                let (a, qs): (&str, &[&str]) = match kind {
                    1 => ("start", &["start the vacuum", "begin vacuuming", "run the robot vacuum", "start cleaning the floors"]),
                    2 => ("stop", &["stop the vacuum", "halt the robot vacuum", "stop vacuuming"]),
                    _ => ("dock", &["send the vacuum back to its dock", "dock the robot vacuum", "send the vacuum home to charge"]),
                };
                let q = rng.pick(qs).to_string();
                (q, vec![call("start_robot_vacuum", json!({"action": a}))], format!("vacuum action {a}."))
            }
        }
    }
}

/// One request that must return no call.
fn refusal(rng: &mut Rng) -> (String, String) {
    match rng.below(5) {
        0 => {
            let q = rng
                .pick(&[
                    "turn off the lights",
                    "dim the lights",
                    "close the blinds",
                    "switch the fan off",
                    "set the fan to high",
                    "turn the lights blue",
                    "adjust the thermostat",
                    "make it warmer",
                    "open the blinds a bit",
                    "dim the lights to 40 percent",
                    "turn up the heat",
                    "start the fan",
                ])
                .to_string();
            (q, "the room or value is missing; no call.".into())
        }
        1 => {
            let q = rng
                .pick(&[
                    "lock the front door",
                    "what's the weather tomorrow",
                    "order more coffee",
                    "play my workout playlist",
                    "is the garage door open",
                    "call mom",
                    "how much battery does the vacuum have",
                    "set an alarm for 7am",
                    "water the plants",
                    "turn on the tv",
                    "what time is it",
                    "add milk to the shopping list",
                ])
                .to_string();
            (q, "no declared tool covers this request; no call.".into())
        }
        2 => {
            let (room, _) = *rng.pick(&ROOMS);
            let w = room_word(rng, room);
            let q = match rng.below(5) {
                0 => format!("don't turn off the {w} lights"),
                1 => format!("do not open the {w} blinds"),
                2 => format!("please don't dim the {w} lights"),
                3 => "never set the thermostat above 25".to_string(),
                _ => format!("don't start the vacuum in the {w}"),
            };
            (q, "the request is negated; no call.".into())
        }
        3 => {
            let (room, _) = *rng.pick(&ROOMS);
            let w = room_word(rng, room);
            let q = match rng.below(3) {
                0 => format!("dim the {w} lights to {} percent", 101 + rng.below(200)),
                1 => format!("set the thermostat to {} degrees", 31 + rng.below(40)),
                _ => format!("set the thermostat to {} degrees", rng.below(10)),
            };
            (q, "the value is outside the allowed range; no call.".into())
        }
        _ => {
            let q = rng
                .pick(&[
                    "turn on the garage lights",
                    "open the bathroom blinds",
                    "turn on the kitchen fan",
                    "vacuum the study",
                    "turn on the hallway fan",
                ])
                .to_string();
            (q, "that room is not supported for this device; no call.".into())
        }
    }
}

/// `n` smart-home examples as `{system, tools, query, reasoning, answers}`.
pub fn smart_home(env: &Value, n: usize, seed: u64, exclude: &[String]) -> Vec<Map<String, Value>> {
    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xA5A5);
    let exclude: std::collections::HashSet<String> = exclude.iter().map(|q| py_strip(q).to_lowercase()).collect();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(n);
    let mut guard = 0;
    while out.len() < n && guard < n * 50 {
        guard += 1;
        let r = rng.below(100);
        let (query, answers, reasoning) = if r < 60 {
            positive(&mut rng)
        } else if r < 75 {
            let (a, ca, ra) = positive(&mut rng);
            let (b, cb, rb) = positive(&mut rng);
            if ca[0]["name"] == cb[0]["name"] {
                continue;
            }
            let joiner = *rng.pick(&[" and ", ", then ", " and also "]);
            (format!("{a}{joiner}{b}"), [ca, cb].concat(), format!("{ra} {rb}"))
        } else {
            let (q, reason) = refusal(&mut rng);
            (q, vec![], reason)
        };
        let key = query.to_lowercase();
        if exclude.contains(&key) || !seen.insert(key) {
            continue;
        }
        let mut ex = Map::new();
        ex.insert("system".into(), env["system"].clone());
        ex.insert("tools".into(), env["tools"].clone());
        ex.insert("query".into(), Value::String(query));
        ex.insert("reasoning".into(), Value::String(reasoning));
        ex.insert("answers".into(), Value::Array(answers));
        out.push(ex);
    }
    out
}

/// The environments bundled with the package.
pub const ENVIRONMENTS: [(&str, &str); 6] = [
    ("data_capture", include_str!("../environments/data_capture.json")),
    ("kitchen_appliance", include_str!("../environments/kitchen_appliance.json")),
    ("media_player", include_str!("../environments/media_player.json")),
    ("productivity", include_str!("../environments/productivity.json")),
    ("smart_home", include_str!("../environments/smart_home.json")),
    ("wearable", include_str!("../environments/wearable.json")),
];

pub fn environment(name: &str) -> Option<Value> {
    ENVIRONMENTS.iter().find(|(n, _)| *n == name).map(|(_, j)| serde_json::from_str(j).expect("bundled environment JSON"))
}
