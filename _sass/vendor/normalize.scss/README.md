# normalize.scss v0.1.0

Normalize.scss is the SCSS version of [normalize.css](http://necolas.github.io/normalize.css), a customisable CSS file that makes browsers render all elements more consistently and in line with modern standards.

[View the normalize.css test file](http://necolas.github.io/normalize.css/latest/test.html)

## Install

* [npm](http://npmjs.org/): `npm install --save normalize.scss`
* [Component(1)](https://github.com/component/component/): `component install guerrero/normalize.scss`
* [Bower](http://bower.io/): `bower install --save normalize.scss`
* Download: Go to [this link](https://raw.githubusercontent.com/guerrero/normalize.scss/master/normalize.scss), press right-click on the page and choose "Save as..."

No other styles should come before Normalize.scss.

It's recommendable to modify `normalize.scss` to suit it to your project

## What does it do?

* Preserves useful defaults, unlike many CSS resets.
* Normalizes styles for a wide range of elements.
* Corrects bugs and common browser inconsistencies.
* Improves usability with subtle improvements.
* Explains what code does using detailed comments.

## Browser support

* Google Chrome (latest)
* Mozilla Firefox (latest)
* Mozilla Firefox 4
* Opera (latest)
* Apple Safari 6+
* Internet Explorer 8+

[Normalize.css v1 provides legacy browser
support](https://github.com/necolas/normalize.css/tree/v1) (IE 6+, Safari 4+),
but is no longer actively developed.

## Extended details

Additional detail and explanation of the esoteric parts of normalize.css.

#### `pre, code, kbd, samp`

The `font-family: monospace, monospace` hack fixes the inheritance and scaling
of font-size for preformated text. The duplication of `monospace` is
intentional.  [Source](http://en.wikipedia.org/wiki/User:Davidgothberg/Test59).

#### `sub, sup`

Normally, using `sub` or `sup` affects the line-box height of text in all
browsers. [Source](http://gist.github.com/413930).

#### `svg:not(:root)`

Adding `overflow: hidden` fixes IE9's SVG rendering. Earlier versions of IE
don't support SVG, so we can safely use the `:not()` and `:root` selectors that
modern browsers use in the default UA stylesheets to apply this style. [SVG
Mailing List discussion](http://lists.w3.org/Archives/Public/public-svg-wg/2008JulSep/0339.html)

#### `input[type="search"]`

The search input is not fully stylable by default. In Chrome and Safari on
OSX/iOS you can't control `font`, `padding`, `border`, or `background`. In
Chrome and Safari on Windows you can't control `border` properly. It will apply
`border-width` but will only show a border color (which cannot be controlled)
for the outer 1px of that border. Applying `-webkit-appearance: textfield`
addresses these issues without removing the benefits of search inputs (e.g.
showing past searches).

#### `legend`

Adding `border: 0` corrects an IE 8–11 bug where `color` (yes, `color`) is not
inherited by `legend`.

## Acknowledgements

Normalize.scss is a project by [Alex Guerrero](https://github.com/guerrero) based on [normalize.css](http://necolas.github.io/normalize.css) from [Nicolas Gallagher](https://github.com/necolas), co-created with [Jonathan Neal](https://github.com/jonathantneal).
