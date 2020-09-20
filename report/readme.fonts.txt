To install Optima, the default sans-serif font in the template, for LaTeX use:
(assuming you're using TeXlive, or MacTeX for Mac):

    wget http://mirrors.ctan.org/fonts/urw/classico/uop.zip
    sudo unzip uop.zip -d/usr/local/texlive/texmf-local/
    echo "Map uop.map" | sudo tee -a /usr/local/texlive/texmf-local/web2c/updmap.cfg
    sudo -H mktexlsr
    sudo -H updmap-sys
