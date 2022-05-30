<img src="../logos/BayesML_logo.png" width="200">

# vs code数式置換メモ（インライン数式限定）

<div style="text-align:right">
作成：中原
</div>

rst -> md
```
:math:`(.*?)` -> $$$1$
```

md -> rst
```
\$(.*?)\$ -> :math:`$1`
```
