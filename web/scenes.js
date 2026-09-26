// A small world per example that the delivered calls act on, so a typed call
// visibly does something. Each scene: mount(root), apply(call) -> log line,
// hold(call) (escalated: shown but not run), reset().

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

const describe = (c) => `${c.name}(${Object.entries(c.arguments || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})`;

// Shared scene chrome: a stage and a short event log.
class Scene {
  mount(root) {
    this.root = root;
    root.replaceChildren();
    this.stage = el("div", `stage stage-${this.id}`);
    this.log = el("ol", "scene-log");
    root.append(this.stage, this.log);
    this.reset();
  }
  note(text, kind = "") {
    const li = el("li", kind, text);
    this.log.prepend(li);
    while (this.log.children.length > 5) this.log.lastChild.remove();
  }
  hold(call) {
    this.note(`held for a bigger model: ${describe(call)}`, "held");
  }
  reset() {
    this.log.replaceChildren();
    this.init();
    this.draw();
  }
}

// // GAME

class Game extends Scene {
  id = "game";
  W = 9;
  H = 5;
  init() {
    this.hero = { x: 1, y: 2, hp: 2, max: 5 };
    this.lit = false;
    this.foes = {
      goblin: { x: 5, y: 1, hp: 2, glyph: "g" },
      skeleton: { x: 6, y: 3, hp: 2, glyph: "s" },
      dragon: { x: 8, y: 2, hp: 5, glyph: "D" },
    };
    this.say = "";
  }
  draw() {
    this.stage.replaceChildren();
    const grid = el("div", "grid-map" + (this.lit ? " lit" : ""));
    for (let y = 0; y < this.H; y++) {
      for (let x = 0; x < this.W; x++) {
        const cell = el("span", "cell");
        const near = Math.abs(x - this.hero.x) + Math.abs(y - this.hero.y) <= 2;
        if (!this.lit && !near) cell.classList.add("dark");
        if (x === this.hero.x && y === this.hero.y) {
          cell.textContent = "@";
          cell.classList.add("you");
        } else {
          const foe = Object.values(this.foes).find((f) => f.x === x && f.y === y);
          if (foe) {
            cell.textContent = foe.hp > 0 ? foe.glyph : "×";
            cell.classList.add(foe.hp > 0 ? "foe" : "dead");
          } else cell.textContent = "·";
        }
        grid.append(cell);
      }
    }
    const hud = el("div", "hud");
    hud.append(el("span", "", "hp "), el("span", "hearts", "♥".repeat(this.hero.hp) + "♡".repeat(this.hero.max - this.hero.hp)));
    hud.append(el("span", "", this.lit ? "  torch lit" : "  dark"));
    this.stage.append(grid, hud);
    if (this.say) this.stage.append(el("div", "bubble", `@ "${this.say}"`));
  }
  apply(call) {
    const a = call.arguments || {};
    let msg;
    if (call.name === "move") {
      const d = { north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0] }[a.direction] || [0, 0];
      const steps = Math.max(1, Math.min(10, a.steps || 1));
      for (let i = 0; i < steps; i++) {
        const nx = Math.max(0, Math.min(this.W - 1, this.hero.x + d[0]));
        const ny = Math.max(0, Math.min(this.H - 1, this.hero.y + d[1]));
        if (Object.values(this.foes).some((f) => f.hp > 0 && f.x === nx && f.y === ny)) break;
        this.hero.x = nx;
        this.hero.y = ny;
      }
      msg = `walked ${a.direction}${a.steps ? ` ${a.steps}` : ""}`;
    } else if (call.name === "attack") {
      const foe = this.foes[a.target];
      if (!foe) msg = `no ${a.target} here`;
      else if (foe.hp <= 0) msg = `the ${a.target} is already down`;
      else {
        const dmg = a.weapon === "fireball" ? 3 : 1;
        foe.hp = Math.max(0, foe.hp - dmg);
        msg = `${a.weapon || "fists"} hit the ${a.target}${foe.hp ? ` (${foe.hp} left)` : ", it falls"}`;
      }
    } else if (call.name === "use_item") {
      if (a.item === "health_potion") {
        this.hero.hp = Math.min(this.hero.max, this.hero.hp + 3);
        msg = "drank a potion";
      } else if (a.item === "torch") {
        this.lit = true;
        msg = "the torch flares up";
      } else msg = `used the ${a.item}`;
    } else if (call.name === "speak") {
      this.say = a.line || "";
      msg = "spoke";
    } else msg = `unknown action ${call.name}`;
    this.draw();
    return msg;
  }
}

// // ROBOT

class Robot extends Scene {
  id = "robot";
  init() {
    this.belt = [
      { object: "cup", color: "red" },
      { object: "bottle", color: "blue" },
      { object: "bolt", color: "yellow" },
      { object: "box", color: "green" },
      { object: "cup", color: "blue" },
    ];
    this.held = null;
    this.bins = { left: [], center: [], right: [] };
    this.speed = "normal";
    this.stopped = false;
  }
  part(p, cls = "") {
    const e = el("span", `part ${p.color} ${cls}`, p.object);
    return e;
  }
  draw() {
    this.stage.replaceChildren();
    const top = el("div", "arm-row");
    const grip = el("div", "gripper" + (this.held ? " holding" : ""));
    grip.append(el("span", "label", "gripper"), this.held ? this.part(this.held) : el("span", "empty-slot", "empty"));
    top.append(grip, el("span", `speed ${this.speed}`, `speed: ${this.speed}`));
    if (this.stopped) top.append(el("span", "estop", "E-STOP"));
    const belt = el("div", "belt");
    belt.append(el("span", "label", "belt"));
    for (const p of this.belt) belt.append(this.part(p));
    const bins = el("div", "bins");
    for (const [name, items] of Object.entries(this.bins)) {
      const b = el("div", "bin");
      b.append(el("span", "label", name));
      for (const p of items) b.append(this.part(p));
      bins.append(b);
    }
    this.stage.append(top, belt, bins);
  }
  apply(call) {
    const a = call.arguments || {};
    let msg;
    if (call.name !== "stop" && this.stopped) {
      this.stopped = false;
    }
    if (call.name === "pick") {
      if (this.held) msg = `already holding the ${this.held.color} ${this.held.object}`;
      else {
        const i = this.belt.findIndex((p) => p.object === a.object && (!a.color || p.color === a.color));
        if (i < 0) msg = `no ${a.color ? a.color + " " : ""}${a.object} on the belt`;
        else {
          this.held = this.belt.splice(i, 1)[0];
          msg = `picked the ${this.held.color} ${this.held.object}`;
        }
      }
    } else if (call.name === "drop_in_bin") {
      if (!this.held) msg = "the gripper is empty";
      else {
        this.bins[a.bin].push(this.held);
        msg = `dropped the ${this.held.object} in the ${a.bin} bin`;
        this.held = null;
      }
    } else if (call.name === "set_speed") {
      this.speed = a.speed;
      msg = `speed set to ${a.speed}`;
    } else if (call.name === "stop") {
      this.stopped = true;
      msg = "emergency stop";
    } else msg = `unknown action ${call.name}`;
    this.draw();
    return msg;
  }
}

// // CAR

class Car extends Scene {
  id = "car";
  init() {
    this.temp = { driver: 21, passenger: 21 };
    this.dest = null;
    this.music = null;
    this.calling = null;
  }
  draw() {
    this.stage.replaceChildren();
    const dash = el("div", "dash");
    for (const zone of ["driver", "passenger"]) {
      const t = el("div", "tile");
      t.append(el("span", "label", zone), el("span", "big", `${this.temp[zone]}°`));
      dash.append(t);
    }
    const nav = el("div", "tile wide");
    nav.append(el("span", "label", "navigation"), el("span", "big", this.dest || "—"));
    const music = el("div", "tile wide");
    music.append(el("span", "label", "now playing"), el("span", "big", this.music || "—"));
    const phone = el("div", "tile wide");
    phone.append(el("span", "label", "phone"), el("span", "big" + (this.calling ? " live" : ""), this.calling ? `calling ${this.calling}…` : "—"));
    dash.append(nav, music, phone);
    this.stage.append(dash);
  }
  apply(call) {
    const a = call.arguments || {};
    let msg;
    if (call.name === "set_temperature") {
      const zones = a.zone === "driver" || a.zone === "passenger" ? [a.zone] : ["driver", "passenger"];
      for (const z of zones) this.temp[z] = a.degrees;
      msg = `${zones.join(" and ")} set to ${a.degrees}°`;
    } else if (call.name === "navigate") {
      this.dest = a.destination;
      msg = `routing to ${a.destination}`;
    } else if (call.name === "play_music") {
      this.music = `${a.query ? a.query + " on " : ""}${a.source}`;
      msg = `playing ${this.music}`;
    } else if (call.name === "call") {
      this.calling = a.contact;
      msg = `calling ${a.contact}`;
    } else msg = `unknown action ${call.name}`;
    this.draw();
    return msg;
  }
}

// // EXPENSES

class Ledger extends Scene {
  id = "expense";
  init() {
    this.rows = [];
  }
  draw() {
    this.stage.replaceChildren();
    const table = el("table", "ledger");
    const head = el("tr");
    for (const h of ["merchant", "category", "amount"]) head.append(el("th", "", h));
    table.append(head);
    for (const r of this.rows) {
      const tr = el("tr");
      tr.append(el("td", "", r.merchant || "—"), el("td", "", r.category || "—"), el("td", "num", `${Number(r.amount).toFixed(2)} ${r.currency}`));
      table.append(tr);
    }
    if (!this.rows.length) {
      const tr = el("tr");
      const td = el("td", "empty-row", "no expenses yet");
      td.colSpan = 3;
      tr.append(td);
      table.append(tr);
    }
    const totals = {};
    for (const r of this.rows) totals[r.currency] = (totals[r.currency] || 0) + Number(r.amount);
    const foot = el("div", "totals", Object.entries(totals).map(([c, v]) => `${v.toFixed(2)} ${c}`).join("  ·  ") || "");
    this.stage.append(table, foot);
  }
  apply(call) {
    if (call.name !== "log_expense") return `unknown action ${call.name}`;
    const a = call.arguments || {};
    this.rows.push(a);
    this.draw();
    return `logged ${a.amount} ${a.currency}${a.category ? ` as ${a.category}` : ""}`;
  }
}

// Custom tools: no world to act on, so the scene just lists the calls.
class Plain extends Scene {
  id = "custom";
  init() {}
  draw() {
    this.stage.replaceChildren(el("div", "plain", "Custom tools have no scene; delivered calls are listed below."));
  }
  apply(call) {
    return `called ${describe(call)}`;
  }
}

export function sceneFor(id) {
  return { game: new Game(), robot: new Robot(), car: new Car(), expense: new Ledger() }[id] || new Plain();
}
