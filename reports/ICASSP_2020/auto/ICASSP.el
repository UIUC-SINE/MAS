(TeX-add-style-hook
 "ICASSP"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "spconf"
    "amsmath"
    "amssymb"
    "graphicx")
   (TeX-add-symbols
    '("norm" 1)
    '("mat" 1)
    "R"
    "vx"
    "vy"
    "vw"
    "vu"
    "vA"
    "vD"
    "argmin"
    "argmax")
   (LaTeX-add-labels
    "sec:intro"
    "fig:diff_lens"
    "fig:pssi_drawing"
    "sec:format"
    "eq:fwd"
    "fig:meas"
    "eq:mtx-vec"
    "sec:pagestyle"
    "sec:typestyle"
    "sec:majhead"
    "ssec:subhead"
    "sssec:subsubhead"
    "sec:print"
    "sec:page"
    "sec:illust"
    "sec:foot"
    "fig:res"
    "sec:prior"
    "sec:refs")
   (LaTeX-add-bibliographies
    "bibliography"))
 :latex)

