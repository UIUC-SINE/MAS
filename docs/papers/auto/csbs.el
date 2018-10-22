(TeX-add-style-hook
 "csbs"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "letterpaper" "margin=1in")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"
    "geometry"
    "cite")
   (LaTeX-add-bibliographies))
 :latex)

