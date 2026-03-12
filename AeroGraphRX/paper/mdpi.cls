%% MDPI LaTeX Class File
%% mdpi.cls
%%
%% A class file for formatting papers for MDPI journals
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mdpi}[2025/03/11 MDPI Journal Class]

% ──────────────────────────────────────────────────────────────
% Class Options - Simple boolean switches
% ──────────────────────────────────────────────────────────────

% Default values
\newif\if@submit
\newif\if@accept
\@submittrue

\DeclareOption{sensors}{\gdef\@journal{sensors}}
\DeclareOption{materials}{\gdef\@journal{materials}}
\DeclareOption{electronics}{\gdef\@journal{electronics}}
\DeclareOption{molecules}{\gdef\@journal{molecules}}
\DeclareOption{applsci}{\gdef\@journal{applsci}}
\DeclareOption{article}{}
\DeclareOption{submit}{\@submittrue\@acceptfalse}
\DeclareOption{accept}{\@accepttrue\@submitfalse}
\DeclareOption{pdftex}{}

\ProcessOptions\relax

% ──────────────────────────────────────────────────────────────
% Load article class
% ──────────────────────────────────────────────────────────────
\PassOptionsToClass{10pt,oneside,onecolumn}{article}
\LoadClass{article}

% ──────────────────────────────────────────────────────────────
% Required Packages
% ──────────────────────────────────────────────────────────────
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsfonts}
\RequirePackage{graphicx}
\RequirePackage{textcomp}
\RequirePackage{xcolor}
\RequirePackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{bm}
\RequirePackage{mathtools}
\RequirePackage{float}
\RequirePackage{caption}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{listings}
\RequirePackage{lineno}
\RequirePackage{amsthm}
\RequirePackage{url}
\RequirePackage{geometry}
\RequirePackage{fancyhdr}

% ──────────────────────────────────────────────────────────────
% Font Setup - Try newpxtext, fallback to times
% ──────────────────────────────────────────────────────────────
\IfFileExists{newpxtext.sty}{%
  \RequirePackage{newpxtext}
  \RequirePackage{newpxmath}
}{%
  \RequirePackage{times}
}

% ──────────────────────────────────────────────────────────────
% Page Geometry - MDPI Specifications (A4)
% ──────────────────────────────────────────────────────────────
\geometry{
  a4paper,
  left=25.65mm,
  right=25.65mm,
  top=23mm,
  bottom=25mm,
  headheight=14.49998pt,
  headsep=10pt
}

% ──────────────────────────────────────────────────────────────
% Header and Footer with fancy page style
% ──────────────────────────────────────────────────────────────
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Sensors}
\fancyhead[R]{\small Article}
\fancyfoot[C]{\small \thepage}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0pt}

% ──────────────────────────────────────────────────────────────
% Section Formatting
% ──────────────────────────────────────────────────────────────
\titleformat{\section}{\large\bfseries}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}

\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.5em}{}
\titlespacing*{\subsection}{0pt}{10pt}{5pt}

\titleformat{\subsubsection}{\normalsize\itshape}{\thesubsubsection}{0.5em}{}
\titlespacing*{\subsubsection}{0pt}{8pt}{4pt}

% ──────────────────────────────────────────────────────────────
% Title, Author, and Metadata Commands
% ──────────────────────────────────────────────────────────────
\newcommand{\Title}[1]{\gdef\@mdpi@title{#1}}
\newcommand{\Author}[1]{\gdef\@mdpi@authors{#1}}
\newcommand{\address}[1]{\gdef\@mdpi@address{#1}}
\newcommand{\corres}[1]{\gdef\@mdpi@corres{#1}}
\newcommand{\received}[1]{\gdef\@mdpi@received{#1}}
\newcommand{\revised}[1]{\gdef\@mdpi@revised{#1}}
\newcommand{\accepted}[1]{\gdef\@mdpi@accepted{#1}}
\newcommand{\published}[1]{\gdef\@mdpi@published{#1}}

\gdef\@mdpi@title{}
\gdef\@mdpi@authors{}
\gdef\@mdpi@address{}
\gdef\@mdpi@corres{}
\gdef\@mdpi@received{}
\gdef\@mdpi@revised{}
\gdef\@mdpi@accepted{}
\gdef\@mdpi@published{}

% ──────────────────────────────────────────────────────────────
% Abstract and Keywords Environments
% ──────────────────────────────────────────────────────────────
\newenvironment{MDPIabstract}{%
  \noindent\textbf{Abstract:}\;\ignorespaces
}{\par}

\newcommand{\keywords}[1]{%
  \vspace{6pt}
  \noindent\textbf{Keywords:}\;\textit{#1}\par
}

% ──────────────────────────────────────────────────────────────
% Theorem-like Environments
% ──────────────────────────────────────────────────────────────
\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\newtheorem{remark}{Remark}

% ──────────────────────────────────────────────────────────────
% Caption Settings
% ──────────────────────────────────────────────────────────────
\captionsetup{
  font=small,
  labelsep=period,
  justification=justified
}

% ──────────────────────────────────────────────────────────────
% Code Listing Style
% ──────────────────────────────────────────────────────────────
\lstdefinestyle{algo}{
  basicstyle=\footnotesize\ttfamily,
  frame=single,
  numbers=left,
  numberstyle=\tiny,
  numbersep=5pt,
  xleftmargin=15pt,
  framexleftmargin=15pt,
  breaklines=true,
  columns=flexible
}

% ──────────────────────────────────────────────────────────────
% Line Numbers for Peer Review (enabled by default for submit mode)
% ──────────────────────────────────────────────────────────────
\if@submit
  \linenumbers
\fi

% ──────────────────────────────────────────────────────────────
% Custom Maketitle for MDPI Format
% ──────────────────────────────────────────────────────────────
\renewcommand{\maketitle}{%
  \begin{center}
    \vspace*{-15pt}
    {\small Article}
    \vspace{10pt}

    {\LARGE\bfseries \@mdpi@title\par}
    \vspace{12pt}

    {\large \@mdpi@authors\par}
    \vspace{8pt}

    {\small\itshape \@mdpi@address\par}
    \vspace{6pt}

    \ifx\@mdpi@corres\@empty\else
      {\small \@mdpi@corres\par}
      \vspace{6pt}
    \fi

    \rule{\textwidth}{0.5pt}
    \vspace{8pt}

    {\small
      \ifx\@mdpi@received\@empty\else
        \textbf{Received:} \@mdpi@received \quad
      \fi
      \ifx\@mdpi@revised\@empty\else
        \textbf{Revised:} \@mdpi@revised \quad
      \fi
      \ifx\@mdpi@accepted\@empty\else
        \textbf{Accepted:} \@mdpi@accepted \quad
      \fi
      \ifx\@mdpi@published\@empty\else
        \textbf{Published:} \@mdpi@published
      \fi
    }

    \vspace{8pt}
    \rule{\textwidth}{0.5pt}
    \vspace{12pt}
  \end{center}
}

% ──────────────────────────────────────────────────────────────
% Unnumbered Sections use default formatting
% ──────────────────────────────────────────────────────────────

% ──────────────────────────────────────────────────────────────
% Bibliography Numbering Style
% ──────────────────────────────────────────────────────────────
\renewcommand{\@biblabel}[1]{[#1]}

% ──────────────────────────────────────────────────────────────
% Text Formatting Defaults
% ──────────────────────────────────────────────────────────────
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\baselineskip}{11pt}
\raggedbottom

\endinput
