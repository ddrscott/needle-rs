//! The schema compiler's regex dialect (`pattern` and `format`): a small
//! ECMAScript-like syntax compiled to a Thompson program and run with
//! search semantics over code points (`FUN_00032c68`, `FUN_00034568`,
//! `FUN_00034768`, `FUN_000737bc`).
//!
//! A pattern that does not parse, or whose program would pass 512
//! instructions (the match instruction included), is dropped by the caller
//! and constrains nothing.

const MAX_STATES: usize = 512;
const MAX_REPEAT: u32 = 64;

#[derive(Clone, Debug)]
enum Ast {
    Empty,
    Char(u32),
    /// Inclusive ranges, possibly negated.
    Class(bool, Vec<(u32, u32)>),
    Dot,
    Start,
    End,
    Concat(Vec<Ast>),
    Alt(Box<Ast>, Box<Ast>),
    /// `max` of `None` is unbounded.
    Repeat(Box<Ast>, u32, Option<u32>),
}

#[derive(Clone, Debug, PartialEq)]
enum Inst {
    Char(u32),
    Class(bool, Vec<(u32, u32)>),
    Dot,
    Start,
    End,
    Split(usize, usize),
    Jmp(usize),
    Match,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Regex {
    prog: Vec<Inst>,
}

const DIGIT: &[(u32, u32)] = &[(0x30, 0x39)];
const WORD: &[(u32, u32)] = &[(0x30, 0x39), (0x41, 0x5a), (0x5f, 0x5f), (0x61, 0x7a)];
const SPACE: &[(u32, u32)] = &[(0x09, 0x0d), (0x20, 0x20)];

/// `FUN_000342b0`: UTF-8 to code points; a byte that does not start a
/// valid sequence stands for itself.
fn code_points(b: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let (len, init) = match c {
            0x00..=0x7f => (1, c as u32),
            0xc0..=0xdf => (2, (c & 0x1f) as u32),
            0xe0..=0xef => (3, (c & 0x0f) as u32),
            0xf0..=0xf7 => (4, (c & 0x07) as u32),
            _ => (0, c as u32),
        };
        if len <= 1 || i + len > b.len() || !b[i + 1..i + len].iter().all(|&x| x & 0xc0 == 0x80) {
            out.push(c as u32);
            i += 1;
            continue;
        }
        let mut cp = init;
        for &x in &b[i + 1..i + len] {
            cp = (cp << 6) | (x & 0x3f) as u32;
        }
        out.push(cp);
        i += len;
    }
    out
}

struct Parser {
    s: Vec<u32>,
    pos: usize,
}

type P<T> = Result<T, ()>;

impl Parser {
    fn peek(&self) -> Option<u32> {
        self.s.get(self.pos).copied()
    }

    fn peek_at(&self, k: usize) -> Option<u32> {
        self.s.get(self.pos + k).copied()
    }

    fn alt(&mut self) -> P<Ast> {
        let mut left = self.seq()?;
        while self.peek() == Some('|' as u32) {
            self.pos += 1;
            let right = self.seq()?;
            left = Ast::Alt(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn seq(&mut self) -> P<Ast> {
        let mut items = vec![];
        while let Some(c) = self.peek() {
            if c == '|' as u32 || c == ')' as u32 {
                break;
            }
            let mut atom = self.atom()?;
            while let Some((min, max)) = self.quantifier()? {
                atom = Ast::Repeat(Box::new(atom), min, max);
            }
            items.push(atom);
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.pop().unwrap_or(Ast::Empty),
            _ => Ast::Concat(items),
        })
    }

    /// A quantifier after an atom, if one follows; a lazy `?` after it is
    /// read and ignored.
    fn quantifier(&mut self) -> P<Option<(u32, Option<u32>)>> {
        let q = match self.peek().and_then(char::from_u32) {
            Some('*') => {
                self.pos += 1;
                (0, None)
            }
            Some('+') => {
                self.pos += 1;
                (1, None)
            }
            Some('?') => {
                self.pos += 1;
                (0, Some(1))
            }
            Some('{') => {
                let save = self.pos;
                self.pos += 1;
                let Some(n) = self.number() else {
                    self.pos = save;
                    return Ok(None);
                };
                let q = match self.peek().and_then(char::from_u32) {
                    Some('}') => {
                        self.pos += 1;
                        (n, Some(n))
                    }
                    Some(',') => {
                        self.pos += 1;
                        if self.peek() == Some('}' as u32) {
                            self.pos += 1;
                            (n, None)
                        } else {
                            let Some(m) = self.number() else {
                                self.pos = save;
                                return Ok(None);
                            };
                            if self.peek() != Some('}' as u32) {
                                self.pos = save;
                                return Ok(None);
                            }
                            self.pos += 1;
                            if m < n || m > MAX_REPEAT {
                                return Err(());
                            }
                            (n, Some(m))
                        }
                    }
                    _ => {
                        self.pos = save;
                        return Ok(None);
                    }
                };
                if q.0 > MAX_REPEAT {
                    return Err(());
                }
                q
            }
            _ => return Ok(None),
        };
        if self.peek() == Some('?' as u32) {
            self.pos += 1;
        }
        Ok(Some(q))
    }

    fn number(&mut self) -> Option<u32> {
        let start = self.pos;
        let mut v: u64 = 0;
        while let Some(d) = self.peek().filter(|c| (0x30..=0x39).contains(c)) {
            v = (v * 10 + (d - 0x30) as u64).min(u32::MAX as u64);
            self.pos += 1;
        }
        (self.pos > start).then_some(v as u32)
    }

    fn atom(&mut self) -> P<Ast> {
        let c = self.peek().ok_or(())?;
        self.pos += 1;
        match char::from_u32(c) {
            Some('(') => {
                if self.peek() == Some('?' as u32) {
                    if self.peek_at(1) != Some(':' as u32) {
                        return Err(());
                    }
                    self.pos += 2;
                }
                let inner = self.alt()?;
                if self.peek() != Some(')' as u32) {
                    return Err(());
                }
                self.pos += 1;
                Ok(inner)
            }
            Some('*' | '+' | '?' | ')') => Err(()),
            Some('[') => self.class(),
            Some('.') => Ok(Ast::Dot),
            Some('^') => Ok(Ast::Start),
            Some('$') => Ok(Ast::End),
            Some('\\') => {
                let e = self.peek().ok_or(())?;
                self.pos += 1;
                match char::from_u32(e) {
                    Some('d') => Ok(Ast::Class(false, DIGIT.to_vec())),
                    Some('w') => Ok(Ast::Class(false, WORD.to_vec())),
                    Some('s') => Ok(Ast::Class(false, SPACE.to_vec())),
                    Some('D') => Ok(Ast::Class(true, DIGIT.to_vec())),
                    Some('W') => Ok(Ast::Class(true, WORD.to_vec())),
                    Some('S') => Ok(Ast::Class(true, SPACE.to_vec())),
                    _ => Ok(Ast::Char(char_escape(e)?)),
                }
            }
            _ => Ok(Ast::Char(c)),
        }
    }

    fn class(&mut self) -> P<Ast> {
        let neg = self.peek() == Some('^' as u32);
        if neg {
            self.pos += 1;
        }
        let mut ranges = vec![];
        let mut first = true;
        loop {
            let c = self.peek().ok_or(())?;
            if c == ']' as u32 && !first {
                self.pos += 1;
                break;
            }
            first = false;
            self.pos += 1;
            let lo = if c == '\\' as u32 {
                let e = self.peek().ok_or(())?;
                self.pos += 1;
                match char::from_u32(e) {
                    Some('d') => {
                        ranges.extend_from_slice(DIGIT);
                        continue;
                    }
                    Some('w') => {
                        ranges.extend_from_slice(WORD);
                        continue;
                    }
                    Some('s') => {
                        ranges.extend_from_slice(SPACE);
                        continue;
                    }
                    Some('D' | 'W' | 'S') => return Err(()),
                    _ => char_escape(e)?,
                }
            } else {
                c
            };
            if self.peek() == Some('-' as u32) && self.peek_at(1).is_some_and(|n| n != ']' as u32) {
                self.pos += 1;
                let e = self.peek().ok_or(())?;
                self.pos += 1;
                let hi = if e == '\\' as u32 {
                    let x = self.peek().ok_or(())?;
                    self.pos += 1;
                    char_escape(x)?
                } else {
                    e
                };
                if hi < lo {
                    return Err(());
                }
                ranges.push((lo, hi));
            } else {
                ranges.push((lo, lo));
            }
        }
        if ranges.is_empty() {
            return Err(());
        }
        Ok(Ast::Class(neg, ranges))
    }
}

/// A single-character escape: `\f \n \r \t \v` or escaped punctuation.
/// Any other escaped letter or digit is an error.
fn char_escape(e: u32) -> P<u32> {
    Ok(match char::from_u32(e) {
        Some('f') => 0x0c,
        Some('n') => 0x0a,
        Some('r') => 0x0d,
        Some('t') => 0x09,
        Some('v') => 0x0b,
        Some(c) if c.is_ascii_alphanumeric() => return Err(()),
        _ => e,
    })
}

struct Emitter {
    prog: Vec<Inst>,
    over: bool,
}

impl Emitter {
    fn push(&mut self, i: Inst) -> usize {
        self.prog.push(i);
        if self.prog.len() > MAX_STATES {
            self.over = true;
        }
        self.prog.len() - 1
    }

    fn emit(&mut self, a: &Ast) {
        if self.over {
            return;
        }
        match a {
            Ast::Empty => {}
            Ast::Char(c) => {
                self.push(Inst::Char(*c));
            }
            Ast::Class(n, r) => {
                self.push(Inst::Class(*n, r.clone()));
            }
            Ast::Dot => {
                self.push(Inst::Dot);
            }
            Ast::Start => {
                self.push(Inst::Start);
            }
            Ast::End => {
                self.push(Inst::End);
            }
            Ast::Concat(items) => items.iter().for_each(|x| self.emit(x)),
            Ast::Alt(l, r) => {
                let split = self.push(Inst::Split(0, 0));
                self.emit(l);
                let jmp = self.push(Inst::Jmp(0));
                let right = self.prog.len();
                self.emit(r);
                let end = self.prog.len();
                if !self.over {
                    self.prog[split] = Inst::Split(split + 1, right);
                    self.prog[jmp] = Inst::Jmp(end);
                }
            }
            Ast::Repeat(x, min, max) => {
                for _ in 0..*min {
                    if self.over {
                        return;
                    }
                    self.emit(x);
                }
                match max {
                    None => {
                        let split = self.push(Inst::Split(0, 0));
                        self.emit(x);
                        let jmp = self.push(Inst::Jmp(split));
                        if !self.over {
                            self.prog[split] = Inst::Split(split + 1, jmp + 1);
                        }
                    }
                    Some(max) => {
                        let mut splits = vec![];
                        for _ in *min..*max {
                            if self.over {
                                return;
                            }
                            splits.push(self.push(Inst::Split(0, 0)));
                            self.emit(x);
                        }
                        let end = self.prog.len();
                        if !self.over {
                            for s in splits {
                                self.prog[s] = Inst::Split(s + 1, end);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Regex {
    pub fn new(pattern: &[u8]) -> Option<Regex> {
        let mut p = Parser { s: code_points(pattern), pos: 0 };
        let ast = p.alt().ok()?;
        if p.pos != p.s.len() {
            return None;
        }
        let mut e = Emitter { prog: vec![], over: false };
        e.emit(&ast);
        e.push(Inst::Match);
        (!e.over).then_some(Regex { prog: e.prog })
    }

    /// Does the pattern match anywhere in `text` (UTF-8 bytes)?
    pub fn is_match(&self, text: &[u8]) -> bool {
        let s = code_points(text);
        let n = self.prog.len();
        let mut cur: Vec<usize> = Vec::with_capacity(n);
        let mut next: Vec<usize> = Vec::with_capacity(n);
        let mut mark = vec![usize::MAX; n];
        for i in 0..=s.len() {
            // Seed a new thread at every position.
            if self.add(&mut cur, &mut mark, 0, i, s.len()) {
                return true;
            }
            if i == s.len() {
                break;
            }
            let c = s[i];
            next.clear();
            let gen_mark = i + 1;
            let mut hit = false;
            let threads = std::mem::take(&mut cur);
            for &pc in &threads {
                let ok = match &self.prog[pc] {
                    Inst::Char(x) => *x == c,
                    Inst::Class(neg, r) => r.iter().any(|&(a, b)| a <= c && c <= b) != *neg,
                    Inst::Dot => !matches!(c, 0x0a | 0x0d | 0x2028 | 0x2029),
                    _ => false,
                };
                if ok && self.add_gen(&mut next, &mut mark, pc + 1, gen_mark, s.len()) {
                    hit = true;
                }
            }
            if hit {
                return true;
            }
            cur = std::mem::take(&mut next);
        }
        false
    }

    fn add(&self, list: &mut Vec<usize>, mark: &mut [usize], pc: usize, pos: usize, len: usize) -> bool {
        self.add_gen(list, mark, pc, pos, len)
    }

    /// Follow the empty moves from `pc` at position `pos`, adding the
    /// consuming instructions to `list`; true when `Match` is reachable.
    fn add_gen(&self, list: &mut Vec<usize>, mark: &mut [usize], pc: usize, pos: usize, len: usize) -> bool {
        if mark[pc] == pos {
            return false;
        }
        mark[pc] = pos;
        match &self.prog[pc] {
            Inst::Match => true,
            Inst::Jmp(x) => self.add_gen(list, mark, *x, pos, len),
            Inst::Split(a, b) => {
                let x = self.add_gen(list, mark, *a, pos, len);
                self.add_gen(list, mark, *b, pos, len) || x
            }
            Inst::Start => pos == 0 && self.add_gen(list, mark, pc + 1, pos, len),
            Inst::End => pos == len && self.add_gen(list, mark, pc + 1, pos, len),
            _ => {
                list.push(pc);
                false
            }
        }
    }
}

/// The regex a known `format` adds (`FUN_00033050`).
pub fn format_pattern(format: &[u8]) -> Option<&'static str> {
    Some(match format {
        b"uuid" => "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        b"date" => "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
        b"time" => "^[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?([Zz]|[+-][0-9]{2}:[0-9]{2})?$",
        b"date-time" => "^[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?([Zz]|[+-][0-9]{2}:[0-9]{2})$",
        b"email" => "^[^@ ]+@[^@ ]+[.][^@ ]+$",
        b"ipv4" => "^[0-9]{1,3}([.][0-9]{1,3}){3}$",
        b"ipv6" => "^[0-9A-Fa-f.]*:[0-9A-Fa-f.:]*$",
        b"duration" => "^P([0-9]+W|([0-9]+Y)?([0-9]+M)?([0-9]+D)?(T([0-9]+H)?([0-9]+M)?([0-9]+([.][0-9]+)?S)?)?)$",
        b"hostname" => "^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?([.][A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(p: &str, t: &str) -> bool {
        Regex::new(p.as_bytes()).unwrap().is_match(t.as_bytes())
    }

    #[test]
    fn search_semantics_and_anchors() {
        assert!(m("ab", "xxabyy"));
        assert!(!m("^ab", "xab"));
        assert!(m("^ab$", "ab"));
        assert!(m("a|b", "zzb"));
        assert!(m("^(?:ab)+$", "ababab"));
        assert!(!m("^a{2,3}$", "aaaa"));
        assert!(m("^a{2,3}$", "aaa"));
        assert!(m("^[^@ ]+@[^@ ]+[.][^@ ]+$", "a@b.co"));
        assert!(m("^\\d+\\.\\d$", "12.5"));
        assert!(m("^.$", "é"));
        assert!(!m("^.$", "\n"));
        assert!(m("^[]a]+$", "]a"));
        assert!(m("^a{x$", "a{x"));
        assert!(m("", "anything"));
    }

    #[test]
    fn rejects_what_the_library_rejects() {
        for p in ["(?=a)", "\\b", "\\1", "a{65}", "a{3,2}", "*a", "[\\D]", "[a", "(a", "a)", "[z-a]", "\\x41"] {
            assert!(Regex::new(p.as_bytes()).is_none(), "{p}");
        }
    }

    #[test]
    fn formats_compile() {
        for f in ["uuid", "date", "time", "date-time", "email", "ipv4", "ipv6", "duration", "hostname"] {
            assert!(Regex::new(format_pattern(f.as_bytes()).unwrap().as_bytes()).is_some(), "{f}");
        }
        let dt = Regex::new(format_pattern(b"date-time").unwrap().as_bytes()).unwrap();
        assert!(dt.is_match(b"2026-09-26T10:00:00Z"));
        assert!(!dt.is_match(b"2026-09-26T10:00:00"));
        assert!(Regex::new(b"^[a-z]{64}[a-z]{64}[a-z]{64}[a-z]{64}[a-z]{64}[a-z]{64}[a-z]{64}[a-z]{64}$").is_none());
    }
}
