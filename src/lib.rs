#![feature(bool_to_option)]

use std::{marker::PhantomData, mem::MaybeUninit};

#[derive(Debug)]
pub struct ParseError {}

pub type ParseResult<'a, Output> = Result<(Output, &'a str), ParseError>;

/// A trait implemented by the various different parsers.
///
/// Parsers take a string as input and attempts to parse it into a value of type `Output`.
pub trait Parser<'a, Output>: Sized {
    /// Parses the input with this parser, yielding any tokens or errors encountered.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Parses one or more whitespaces.
    /// whitespace1().parse("     d")?; // Returns: ((), "d")
    /// ```
    ///
    /// # Errors
    ///
    /// If the parser is unable to parse the input, one of the [ParseError] variants will be returned.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing the parsed output and the remainding input.
    fn parse(&self, input: &'a str) -> ParseResult<'a, Output>;

    /// Ignores the returned output and yields `()` instead.
    ///
    /// This is useful for transforming a long sequence of parsers into a simple output type
    /// when the returned output is not important for the parser.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let line = literal("#").take_until(|c| == '\n').ignored(); // Output is `()`
    /// ```
    fn ignored(self) -> Ignored<Self, Output> {
        To(self, (), PhantomData)
    }

    /// Maps `self` to `U` by applying F.
    fn map<U, F: Fn(Output) -> U>(self, f: F) -> Map<Self, F, Output> {
        Map(self, f, PhantomData)
    }

    /// Parses with `self` or `other` if parsing with `self` failed.
    fn or<P: Parser<'a, Output>>(self, other: P) -> Or<Self, P> {
        Or(self, other)
    }

    /// Parses optionally with `self`. Otherwise returns the input just as it was received.
    ///
    /// # Returns
    ///
    /// Returns a parser yielding `(Option<Output>, &str)`.
    /// If the parsing was successfull, the option is [Some]. Otherwise it is [None].
    /// This parser is unable to return an error.
    fn or_not(self) -> OrNot<Self> {
        OrNot(self)
    }

    /// Repeats parsing with `self`, `N` times.
    ///
    /// # Errors
    ///
    /// Returns a [ParseError] variant if any error is found while parsing,
    /// aborting any iterations it may have left.
    ///
    /// # Returns
    ///
    /// Returns a parser yielding `[Output; N]`.
    fn repeat<const N: usize>(self) -> Repeat<Self, N> {
        Repeat(self)
    }

    /// Chains two parses, yielding `(Output, U)` if successful.
    fn then<U, P: Parser<'a, U>>(self, other: P) -> Then<Self, P> {
        Then(self, other)
    }

    /// Parses with `self` and sets the output to `x`.
    fn to<U>(self, x: U) -> To<Self, Output, U> {
        To(self, x, PhantomData)
    }
}

// Implements [Parser] for any function that takes an input and returns [ParseResult].
impl<'a, Output, F: Fn(&'a str) -> ParseResult<'a, Output> + Clone> Parser<'a, Output> for F {
    // TODO: God til rapport.
    fn parse(&self, input: &'a str) -> ParseResult<'a, Output> {
        self(input)
    }
}

pub type Ignored<P, O> = To<P, O, ()>;

#[derive(Clone)]
pub struct Map<A, F, O>(A, F, PhantomData<O>);

impl<'a, O, A: Parser<'a, O>, U, F: Fn(O) -> U> Parser<'a, U> for Map<A, F, O> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, U> {
        self.0.parse(input).map(|(o, rest)| (self.1(o), rest))
    }
}

#[derive(Clone)]
pub struct Or<A, B>(A, B);

impl<'a, O, A: Parser<'a, O>, B: Parser<'a, O>> Parser<'a, O> for Or<A, B> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, O> {
        match self.0.parse(input) {
            Ok(o) => Ok(o),
            _ => self.1.parse(input),
        }
    }
}

#[derive(Clone, Copy)]
pub struct OrNot<A>(A);

impl<'a, O, A: Parser<'a, O>> Parser<'a, Option<O>> for OrNot<A> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, Option<O>> {
        let result = self.0.parse(input);
        result.map_or(Ok((None, input)), |(o, rest)| Ok((Some(o), rest)))
    }
}

#[derive(Clone, Copy)]
pub struct Repeat<A, const N: usize>(A);

trait RepeatInit<O> {
    const INIT: MaybeUninit<O> = MaybeUninit::uninit();
}

impl<'a, O, A: Parser<'a, O>, const N: usize> RepeatInit<O> for Repeat<A, N> {}

impl<'a, O, A: Parser<'a, O>, const N: usize> Parser<'a, [O; N]> for Repeat<A, N> {
    fn parse(&self, mut input: &'a str) -> ParseResult<'a, [O; N]> {
        let mut output: [MaybeUninit<O>; N] = [Self::INIT; N];
        for i in 0..N {
            let (o, rest) = self.0.parse(input)?;
            input = rest;
            output[i] = MaybeUninit::new(o);
        }

        Ok((output.map(|item| unsafe { item.assume_init() }), input))
    }
}

#[derive(Clone, Copy)]
pub struct Then<A, B>(A, B);

impl<'a, O, U, A: Parser<'a, O>, B: Parser<'a, U>> Parser<'a, (O, U)> for Then<A, B> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, (O, U)> {
        match self.0.parse(input) {
            Ok((o, rest)) => self.1.parse(rest).map(|(u, rest)| ((o, u), rest)),
            Err(e) => Err(e),
        }
    }
}

#[derive(Clone, Copy)]
pub struct To<A, O, U>(A, U, PhantomData<O>);

impl<'a, O, A: Parser<'a, O>, U: Clone> Parser<'a, U> for To<A, O, U> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, U> {
        self.0.parse(input).map(|(_, rest)| (self.1.clone(), rest))
    }
}

pub trait ParserExt<'a, O, U>: Parser<'a, (O, U)> {
    fn left(self) -> Left<Self, O, U> {
        Left(self, PhantomData)
    }

    fn right(self) -> Right<Self, O, U> {
        Right(self, PhantomData)
    }
}

impl<'a, O, U, P: Parser<'a, (O, U)>> ParserExt<'a, O, U> for P {}

#[derive(Clone, Copy)]
pub struct Left<A, O, U>(A, PhantomData<(O, U)>);

impl<'a, O, U, A: Parser<'a, (O, U)>> Parser<'a, O> for Left<A, O, U> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, O> {
        self.0.parse(input).map(|((left, _), rest)| (left, rest))
    }
}

#[derive(Clone, Copy)]
pub struct Right<A, O, U>(A, PhantomData<(O, U)>);

impl<'a, O, U, A: Parser<'a, (O, U)>> Parser<'a, U> for Right<A, O, U> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, U> {
        self.0.parse(input).map(|((_, right), rest)| (right, rest))
    }
}

#[derive(Clone, Copy)]
pub struct Any;

impl<'a> Parser<'a, char> for Any {
    fn parse(&self, input: &'a str) -> ParseResult<'a, char> {
        let next = input.chars().next();
        next.map_or(Err(ParseError {}), |c| Ok((c, &input[c.len_utf8()..])))
    }
}

/// Parses any character.
pub fn any() -> Any {
    Any {}
}

#[derive(Clone, Copy)]
pub struct Char(char);

impl<'a> Parser<'a, ()> for Char {
    fn parse(&self, input: &'a str) -> ParseResult<'a, ()> {
        match any().parse(input)? {
            (c, rest) if c == self.0 => Ok(((), rest)),
            _ => Err(ParseError {}),
        }
    }
}

/// Parses a single character matching `c`.
pub fn char(c: char) -> Char {
    Char(c)
}

#[derive(Clone, Copy)]
pub struct Empty;

impl<'a> Parser<'a, ()> for Empty {
    fn parse(&self, input: &'a str) -> ParseResult<'a, ()> {
        if input.eq("") {
            return Ok(((), input));
        }

        return Err(ParseError {});
    }
}

/// Parses an empty input eg. `""` as successful.
pub fn empty() -> Empty {
    Empty {}
}

#[derive(Clone, Copy)]
pub struct Literal(&'static str);

impl<'a> Parser<'a, ()> for Literal {
    fn parse(&self, input: &'a str) -> ParseResult<'a, ()> {
        match input.starts_with(self.0) {
            true => Ok(((), &input[self.0.len()..])),
            _ => Err(ParseError {}),
        }
    }
}

pub fn literal(s: &'static str) -> Literal {
    Literal(s)
}

#[derive(Clone, Copy)]
pub struct Take(usize);

impl<'a> Parser<'a, &'a str> for Take {
    fn parse(&self, input: &'a str) -> ParseResult<'a, &'a str> {
        input.get(..self.0).map_or(Err(ParseError {}), |s| {
            Ok((&input[..s.len()], &input[s.len()..]))
        })
    }
}

pub fn take(n: usize) -> Take {
    Take(n)
}

#[derive(Clone, Copy)]
pub struct TakeWhile<F>(F);

impl<'a, F: Fn(char) -> bool> Parser<'a, &'a str> for TakeWhile<F> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, &'a str> {
        let chars = input.chars().enumerate().take_while(|&(_, c)| self.0(c));
        match chars.last() {
            Some((i, _)) => Ok((&input[0..i + 1], &input[i + 1..])),
            _ => Ok((input, input)),
        }
    }
}

pub fn take_while<F: Fn(char) -> bool>(f: F) -> TakeWhile<F> {
    TakeWhile(f)
}

#[derive(Clone, Copy)]
pub struct TakeWhile1<F>(F);

impl<'a, F: Fn(char) -> bool> Parser<'a, &'a str> for TakeWhile1<F> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, &'a str> {
        let chars = input.chars().enumerate().take_while(|&(_, c)| self.0(c));
        match chars.last() {
            Some((i, _)) => Ok((&input[0..i + 1], &input[i + 1..])),
            _ => Err(ParseError {}),
        }
    }
}

pub fn take_while1<F: Fn(char) -> bool>(f: F) -> TakeWhile1<F> {
    TakeWhile1(f)
}

#[derive(Clone, Copy)]
pub struct TakeUntil<F>(F);

impl<'a, F: Fn(char) -> bool> Parser<'a, &'a str> for TakeUntil<F> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, &'a str> {
        let mut chars = input.chars().enumerate();
        while let Some((i, c)) = chars.next() {
            if self.0(c) {
                return Ok((&input[..i + 1], &input[i + 1..]));
            }
        }

        Ok((input, ""))
    }
}

pub fn take_until<F: Fn(char) -> bool>(f: F) -> TakeUntil<F> {
    TakeUntil(f)
}

#[derive(Clone, Copy)]
pub struct ParseWhile<P>(P);

impl<'a, O, P: Parser<'a, O>> Parser<'a, Vec<O>> for ParseWhile<P> {
    fn parse(&self, mut input: &'a str) -> ParseResult<'a, Vec<O>> {
        let mut output = Vec::new();
        loop {
            match self.0.parse(input) {
                Ok((value, rest)) => {
                    input = rest;
                    output.push(value);
                }
                _ => break,
            }
        }

        Ok((output, input))
    }
}

pub fn parse_while<'a, O, P: Parser<'a, O>>(p: P) -> ParseWhile<P> {
    ParseWhile(p)
}

/// Parses zero or more digits with `radix`.
pub fn digits<'a>(radix: u32) -> impl Parser<'a, &'a str> + Clone {
    take_while(move |c| c.is_digit(radix))
}

/// Parses one or more digits with `radix`.
pub fn digits1<'a>(radix: u32) -> impl Parser<'a, &'a str> + Clone {
    take_while1(move |c| c.is_digit(radix))
}

/// Parses zero or more whitespace characters.
pub fn whitespace<'a>() -> impl Parser<'a, ()> + Clone {
    take_while(|c| c.is_whitespace()).ignored()
}

/// Parses one or more whitespace characters.
pub fn whitespace1<'a>() -> impl Parser<'a, ()> + Clone {
    take_while1(|c| c.is_whitespace()).ignored()
}

/// Parses a single i32.
///
/// `int32 = ["-"], {digit};`
pub fn int32<'a>() -> impl Parser<'a, i32> + Clone {
    move |input: &'a str| {
        let (_, rest) = char('-').or_not().then(digits1(10)).parse(input)?;
        Ok((input[..input.len() - rest.len()].parse().unwrap(), rest))
    }
}

/// Parses a single f32.
///
/// `f32 = [-], digit, {digit}, ".", {digit};`
pub fn f32<'a>() -> impl Parser<'a, f32> + Clone {
    move |input: &'a str| {
        let parser = char('-')
            .or_not()
            .then(digits1(10).then(literal(".").then(digits(10)).or_not()));
        let (_, rest) = parser.parse(input)?;
        Ok((input[..input.len() - rest.len()].parse().unwrap(), rest))
    }
}

/// Parses a comment.
///
/// `comment = "#", characters, endOfLine;`
pub fn comment<'a>() -> impl Parser<'a, &'a str> + Clone {
    let parser = literal("#").then(take_until(|c| c == '\n')).right();
    whitespace().then(parser).right()
}

/// A vertex in \[x, y, z, w] format.
pub type Vertex = [f32; 4];

/// Parses a vertex in (x, y, z, \[w]) format where w is optional and defaults to `1.0`.
///
/// `vertex = "v", whitespace, f32, whitespace, f32, whitespace, f32 [,whitespace, f32];`
pub fn vertex<'a>() -> impl Parser<'a, Vertex> {
    let coordinate = whitespace1().then(f32()).right();
    let parser = literal("v").then(coordinate.clone().repeat::<3>()).right();
    let parser = parser.then(coordinate.or_not());
    parser.map(|(left, right)| [left[0], left[1], left[2], right.unwrap_or(1.0)])
}

/// A texture coordinate in \(u, v, w) format; clamped to min 0.0 and max 1.0.
pub type TextureCoordinate = [f32; 3];

/// Parses a  texture coordinate in (u, \[v, w]) format where v, w is optional and defaults to `0.0`.
///
/// `textureCoordinate = "vt", whitespace, f32 [,whitespace, f32 [,whitespace, f32]];`
pub fn texture_coordinate<'a>() -> impl Parser<'a, TextureCoordinate> {
    let coordinate = whitespace1().then(f32()).right();
    let parser = literal("vt").then(coordinate.clone()).right();
    let parser = parser.then(coordinate.clone().then(coordinate.or_not()).or_not());
    parser.map(|(u, vw)| {
        let (v, w) = vw.map_or((0.0, 0.0), |(v, w)| (v, w.unwrap_or(0.0)));
        [u, v, w]
    })
}

/// A vertex normal in (x, y, z) format. These are not required to be unit vectors.
pub type VertexNormal = [f32; 3];

/// Parses a vertex normal in (x, y, z) format.
///
/// `vertexNormal = "vn", whitespace, f32, whitespace, f32, whitespace, f32;`
pub fn vertex_normal<'a>() -> impl Parser<'a, VertexNormal> {
    let coordinate = whitespace1().then(f32()).right();
    literal("v").then(coordinate.repeat::<3>()).right()
}

/// A vertex parameter in (u, [v, w]) format. `w` is unable to be `Some` if `v` is not.
#[derive(Clone, Debug)]
pub struct VertexParameter {
    pub u: f32,
    pub v: Option<f32>,
    pub w: Option<f32>,
}

/// Parses a vertex parameter in (u, \[v, w]) format.
///
/// `vertexParameter = "vp", whitespace, f32 [,whitespace, f32 [,whitespace, f32]];`
fn vertex_parameter<'a>() -> impl Parser<'a, VertexParameter> {
    let coordinate = whitespace1().then(f32()).right();
    let w = coordinate.clone().or_not();
    let v = coordinate.clone().then(w).or_not();
    let u = coordinate.clone().then(v);
    literal("vp").then(u).right().map(|(u, vw)| {
        let (v, w) = vw.map_or((None, None), |(v, w)| (Some(v), w));
        VertexParameter { u, v, w }
    })
}

/// A single face element.
#[derive(Default, Debug, Clone, Copy)]
pub struct FaceElement {
    pub vertex_index: i32,
    pub texture_index: Option<i32>,
    pub normal_index: Option<i32>,
}

impl FaceElement {
    pub fn new(vertex_index: i32, texture_index: Option<i32>, normal_index: Option<i32>) -> Self {
        Self {
            vertex_index,
            texture_index,
            normal_index,
        }
    }
}

/// A sequence of multiple face elements.
pub type Face = Vec<FaceElement>;

/// Parses a face.
///
/// `faceElement = int32, [, ("/", int32 [, "/" int32])] | ("//", int32);`
/// `face = "f", whitespace, faceElement, {whitespace, faceElement};`
pub fn face<'a>() -> impl Parser<'a, Face> {
    fn face_element<'a>() -> impl Parser<'a, FaceElement> + Clone {
        let segment = char('/').then(int32()).right();
        let parser1 = segment.clone().then(segment.or_not());
        let parser1 = parser1.map(|(left, right)| (Some(left), right));
        let parser2 = literal("//").then(int32());
        let parser2 = parser2.map(|(_, right)| (None, Some(right)));
        let parser = int32().then(parser1.or(parser2).or_not());
        parser.map(|(vi, tini)| {
            let (ti, ni) = tini.unwrap_or((None, None));
            FaceElement::new(vi, ti, ni)
        })
    }

    let face_element = whitespace1().then(face_element()).right();
    let face_elements = parse_while(face_element.clone());
    let parser = char('f').then(face_element).right().then(face_elements);
    parser.map(|(value, mut output)| {
        output.insert(0, value);
        output
    })
}

/// A sequence of vertices.
pub type Line = Vec<i32>;

/// Parses a line.
///
/// `line = "l", whitespace int32, {int32, whitespace};`
pub fn line<'a>() -> impl Parser<'a, Line> {
    let element = whitespace1().then(int32()).right();
    let elements = parse_while(element.clone());
    let parser = char('l').then(element).right().then(elements);
    parser.map(|(value, mut output)| {
        output.insert(0, value);
        output
    })
}

#[derive(Clone, Debug)]
pub enum Value {
    Empty,
    Vertex(Vertex),
    TextureCoordinate(TextureCoordinate),
    VertexNormal(VertexNormal),
    VertexParameter(VertexParameter),
    Face(Face),
    Line(Line),
}

pub fn value<'a>() -> impl Parser<'a, (Value, Option<&'a str>)> {
    empty()
        .to(Value::Empty)
        .or(vertex().map(|v| Value::Vertex(v)))
        .or(texture_coordinate().map(|v| Value::TextureCoordinate(v)))
        .or(vertex_normal().map(|v| Value::VertexNormal(v)))
        .or(vertex_parameter().map(|v| Value::VertexParameter(v)))
        .or(face().map(|f| Value::Face(f)))
        .or(line().map(|l| Value::Line(l)))
        .then(comment().or_not())
        .or(comment().map(|s| (Value::Empty, Some(s))))
}

#[derive(Clone, Default, Debug)]
pub struct Obj {
    pub vertices: Vec<Vertex>,
    pub texture_coordinates: Vec<TextureCoordinate>,
    pub vertex_normals: Vec<VertexNormal>,
    pub vertex_parameters: Vec<VertexParameter>,
    pub faces: Vec<Face>,
    pub lines: Vec<Line>,
}

impl Obj {
    pub fn add(&mut self, value: Value) {
        match value {
            Value::Vertex(value) => self.vertices.push(value),
            Value::TextureCoordinate(value) => self.texture_coordinates.push(value),
            Value::VertexNormal(value) => self.vertex_normals.push(value),
            Value::VertexParameter(value) => self.vertex_parameters.push(value),
            Value::Face(value) => self.faces.push(value),
            Value::Line(value) => self.lines.push(value),
            Value::Empty => (),
        };
    }
}

pub fn parse(input: &str) -> Result<Obj, ParseError> {
    let mut obj = Obj::default();
    let mut values = Vec::default();

    for line in input.lines() {
        values.push(value().parse(line)?)
    }

    values.into_iter().for_each(|(v, _)| obj.add(v.0));
    Ok(obj)
}

#[cfg(test)]
mod tests {
    use crate::parse;

    #[test]
    fn integration() {
        parse(include_str!("../obj/cornell_box.obj")).unwrap();
    }
}
