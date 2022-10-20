



| Layer     | $\mathcal{I}.1$                               | $\mathcal{I}.2-6$                    | $\mathcal{I}.7$                        |

|-----------|-----------------------------------------------|--------------------------------------|----------------------------------------|

| Input     | $\desired$                                    | $\mathcal{I}.(l-1)$                  | $\mathcal{I}.6$                        |

| Structure | $\begin{bmatrix}\text{Linear}(K\cdot N, 1000) |

| Output    | $[1000]$                                      | $[1000]$                             | $\xoj[N_e \times M\cdot N]$            |

| Layer     | $\mathcal{\boldsymbol{\eta}}_i$               | $\mathcal{P}$                        | $\mathcal{R}$                          | $\mathcal{N_i}$                      | $\mathcal{E_i}$                   |

| Input     | $\{\ximj\}_j$                                 | $\{\ximj, \mathcal{\eta_i}\}$        | $\xijbar$                              | $\xijtilde$                          | $\xijhat$                         |

| Structure | \Cref{tab:direction_evaluate}                 | \Cref{eq:projection}                 | \Cref{eq:retraction}                   | \Cref{tab:prune}                     | \Cref{tab:expand}                 |

| Output    | $\{\etaij\}_j[N_e \times M\cdot N], \beta_i$  | $\{\xijbar\}_j[N_e \times M\cdot N]$ | $\{\xijtilde\}_j[N_e \times M\cdot N]$ | $\{\xijhat\}_j[N_p \times M\cdot N]$ | $\{\xij\}_j[N_e \times M\cdot N]$ |

| Layer     | $\mathcal{\boldsymbol{\eta}}.1$               | $\mathcal{\boldsymbol{\eta}}.2-3$    | $\mathcal{\boldsymbol{\eta}}.4$        |

| Input     | $\{\ximj\}_j$                                 | $\mathcal{\boldsymbol{\eta}}.(l-1)$  | $\mathcal{\boldsymbol{\eta}}.3$        |

| Structure | $\begin{bmatrix}\text{Linear}(M\cdot N, 1000) |

| Output    | $[1000]$                                      | $[1000]$                             | $\{\etaij\}_j[N_e \times M\cdot N]$    |

| Layer     | $\mathcal{N}.1$                               | $\mathcal{N}.2-3$                    | $\mathcal{N}.4$                        |

| Input     | $\{\xijtilde\}_j$                             | $\mathcal{N}.(l-1)$                  | $\mathcal{N}.3$                        |

| Structure | $\begin{bmatrix}\text{Linear}(M\cdot N, 1000) |

| Output    | $[1000]$                                      | $[1000]$                             | $\{\xijhat\}_j[N_e \times M\cdot N]$   |

| Layer     | $\mathcal{E}.1$                               | $\mathcal{E}.2-3$                    | $\mathcal{E}.4$                        |

| Input     | $\{\xijhat\}_j$                               | $\mathcal{E}.(l-1)$                  | $\mathcal{E}.3$                        |

| Structure | $\begin{bmatrix}\text{Linear}(M\cdot N, 1000) |

| Output    | $[1000]$                                      | $[1000]$                             | $\{\xij\}_j[N_e \times M\cdot N]$      |
