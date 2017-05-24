# ************************************************
# *                   CStat2017                  *
# *          Abgabe von Mihran Hakobyan          *
# *                                              *
# *             Donnerstag, 12 Uhr               *
# ************************************************

## ####### ####### ##### AUFGABE 1
library(klaR)
library(mlbench)
library(plyr)
load("crunchigkeit.RData")
data("Ionosphere")

# crossvalid - bestimmt mittlere Fehlerklassifikationsrate
# einer k-fache Kreuzvalidierung von regularisierter Diskriminanzanalyse
#
# Input: param - numerischer Vektor (Parameter der Diskriminanzanalyse)
#         data - Dataframe oder Matrix (Datensatz)
#          k   - natuerliche Zahl (Parameter der Kreuzvalidierung)
#
# Output:  numerische Zahl (Mittlere Fehlerklassifikationsrate)

crossvalid = function(param, data, k){
  gamma = param[1]
  lambda = param[2]
  obs = nrow(data)
  var = ncol(data)
  group = list()
  true.class = list()
  num.elem = floor(obs / k)
  i = 1
  ind = sample(1:obs)
  while(i <= k){
    group[[i]] = data[ind[(1 + num.elem * (i - 1)):(i * num.elem)], ]
    true.class[[i]] = data[ind[(1 + num.elem * (i - 1)):(i * num.elem)],]$Class
    i = i + 1
  }
  r = obs %% k
  if (r != 0){
    while(r > 0){
      ind = sample(1:k, 1)
      group[[ind]] = rbind(group[[ind]], data[num.elem * k + r, ])
      true.class[[ind]] = as.factor(c(as.character(true.class[[ind]]),
        as.character(data[num.elem * k + r,]$Class)))
      r = r - 1
    }
  }
  misclass.rate = numeric(k)
  model.param = matrix(nrow = k, ncol = 2)
  j = k
  while(j > 0){
    test = group[[j]]
    train = do.call(rbind, group[-j])
    model = rda(Class ~ ., data = train[, 3:var], gamma = gamma, lambda = lambda)
    pred.class = predict(model, newdata = test[, 3:var])$class
    misclass.rate[j] = sum(pred.class != true.class[[j]]) / length(pred.class)
    j = j - 1
  }
  return(mean(misclass.rate))
}

crossvalid(c(1,1), data = Ionosphere, k =  10)

# crossvalid soll nun bzgl. der beiden Parameter optimiert werden
# optim(c(0.4,0.7), crossvalid, data = Ionosphere, k = 10, method = "SANN")
# leider keine Konvergenz

######## ####### #######  AUFGABE 2

########  b)

# subsampling - bestimmt mit der Bootstrap Methode das arithmetische Mittel
# der mittleren quadratische Abweichungen zwischen tatsaechlichen und vorher-
# gesagten Werten einer linearen Regression
#
# Input: data - Dataframe oder Matrix (Datensatz)
#          B   - natuerliche Zahl (Wiederholungszahl der Bootstrap-Methode)
#
# Output: numerische Zahl (Mittel der mittleren quadratsichen Abweichungen)

subsampling = function(data, B){
  N = nrow(data)
  bootstrapOnce  = function(){
    samples = sample(1:N, replace = TRUE)
    samples = unique(samples)
    n = length(samples)
    target.var = names(data)[1]
    model = lm(as.formula(paste(target.var, " ~ . ")), data[samples, ])
    pred = predict(model, data[- samples, ])
    mse = sum((pred - data[- samples, ][1]) ^ 2) / n
  }
  MSE = replicate(B, bootstrapOnce())
  return (sum(MSE) / B)
}

subsampling(swiss,200)

########  a) und c)

# variableSelect - Bestimmt die beste Variablenmenge des linearen
# Regressionsmodells mittels Maximierung des R-Quadrat und
# die beste Variablenmenge mittels Minimierung des Guetemass
# berechnet durch subsampling
#
# Input: data - Dataframe oder Matrix (Datensatz)
#          B  - natuerliche Zahl (Wiederholungszahl der Bootstrap-Methode)
#
# Output: Liste mit beiden Variablenmengen, siehe Funktionskopf

variableSelect = function(data, B){
  # maximale anzahl an predictors
  nmax.pred = ncol(data) - 1
  # Matrix aller moeglichen Variablenmengen
  bitstring = expand.grid(as.data.frame(matrix(rep(c(1, 0), nmax.pred), 2)))
  # alle moeglichen Variablenmengen als Listenelemente (Vektoren)
  var.set = apply(as.matrix(bitstring), 1, function(x) {which(x != 0)})
  # Die leere Variablenmenge wird entfernt
  var.set[length(var.set)] = NULL
  # Modelle zugehoeriger Variablenmengen.
  # c(1, var + 1) da erste Variable Zielvariable
  target.var = names(data)[1]
  MODEL = lapply(var.set, function(var)
    {lm(as.formula(paste(target.var, " ~ . ")) , data[, c(1, var + 1)])})
  adj.r.squared = sapply(MODEL, function(model) {summary(model)$adj.r.squared})
  # Listenelement mit der besten Variablenmenge
  best.lvar = which.max(adj.r.squared)
  # Guetemass fuer jede Variablenmenge bestimmen
  qual.meas = lapply(var.set, function(var) {subsampling(data[, c(1, var + 1)], B)})
  # Listenelement mit der besten Variablenmenge (mit Bootstrap)
  best.lvar.bootstrap = which.min(qual.meas)
  return(list(names(data)[var.set[[best.lvar]] + 1],
    names(data)[var.set[[best.lvar.bootstrap]] + 1]))
}

variableSelect(swiss, 200)



####### d)

# In a) wird die Guete des Modells anhand eines einzigen Modells basierend
# auf dem gesamten Datensatz gemessen.
# In c) hingegen wird die Guete des Modells anhand vieler Modelle
# basierend auf einem Teil des Datensatzes gemesssen.
