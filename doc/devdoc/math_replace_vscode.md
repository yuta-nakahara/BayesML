<img src="../logos/BayesML_logo.png" width="200">

# vs code数式置換メモ Ver.2

<div style="text-align:right">
作成：中原
</div>

## インライン数式

rst -> md
```
:math:`(.*?)` -> $$$1$
```

md -> rst
```
\$(.*?)\$ -> :math:`$1`
```

## .mdファイル中のLaTeX数式のpdf出力用

以下の順序で置換する．`\`がエスケープ文字であること，二つの`_`で挟むことが強調の命令になっていることから，以下の置換をしておかないとpdfに変換した際に数式がうまく表示されない．（VS Codeのプレビュー機能であれば以下の置換をせずに正しく表示される．）

1. `\\ --> \cr`
2. `_ --> \_`
3. `\{ --> \\{`
4. `\} --> \\}`
