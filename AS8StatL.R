# function to train a nueral network with a single hidden layer
my.neuralnet<-function(X1,Y,hidden=3,output="linear"){
  #adds a column of 1s to the input X1
  X<-cbind(1,X1)
  #initializes length of the input 
  input.layer.length<-length(X[1,])
  #initializes weights with random normal values
  w01<-rnorm(input.layer.length*hidden)
  w02<-rnorm(hidden)
  w0<-c(w01,w02)
  print(w0)
  
  #the optimization fuction to optimize the weights using conjugate gradient method
  myoptfunc<-
    function(w0){my.eval2.nnet(w0,X,Y,hidden,output)$llik}
  dum<-optim(w0,myoptfunc,method="CG")
  wfinal<-dum$par
  ss<-dum$val
  #returns predicted values by evaluating inputs using the optimized weights
  pred<-my.eval2.nnet(wfinal,X,Y,hidden,output)$pred
  #plots predicted vs orginal Y values 
  plot(pred,Y)
  #returns list of optim values (ss) and final weights
  list(ss=ss,wfinal=wfinal)
}
#funtion to evaluate neural net mod for a single input and given weight 
my.eval1.nnet<-function(Xrow,w0,hidden,output){
    input.layer.length<-length(Xrow)
    # seperate weights for connections between layers (input, hidden, output)
    w01<-w0[c(1:(length(Xrow)*hidden))]
    w02<-w0[-c(1:(length(Xrow)*hidden))]
    w1<-matrix(w01,input.layer.length,hidden)
    print(w1)
    print(Xrow)
    print(w02)
    
    #calculates the weigted sum of the input 
    xhidden<-t(Xrow)%*%w1
    #apply the activation function (logistic sigmoid)
    zhidden<-my.logistic(xhidden)
    #calculates the weigted sum of the hidden units 
    out<-sum(w02*zhidden)
    #apply the activation func if output is binary 
    if(output=="binary"){
      out<-my.logistic(out)
    }
    out
  }
#function to eval neural net mod for a set of inputs and a given weight vec
my.eval2.nnet<-function(w0,X,Y,hidden,output){
    zfunc<-
      function(V){my.eval1.nnet(V,w0,hidden,output)}
    #apply the model to inputs
    pred<-apply(X,1,zfunc)
    #calculate the loss values (we are trying to minimize these)
    if(output=="binary"){
      llik<-(-1)*sum(log(pred)*Y+log(1-pred)*(1-Y))
    }else{
      llik<-sum((pred-Y)^2)
    }
    #return predicted values and loss values
    list(llik=llik,pred=pred,y=Y)
}
#logistic sigmmoid activation function 
#to allow the networj to learn complex nonlinear relationships between input and output variables
my.logistic<-function(z){exp(z)/(1+exp(z))}

#multilayer with regularization 
#the following code generalizes the code above by adding two new parameters
#1. num.layers which allows the user to specify the number of hidden layers 
#thish means that all of the functions that we used previously have been modfied
#to handle multiple hidden layers.
#2. Lambda which enocourages the optimization function to choose smaller values for the weightes 
#to reduce the risk of over fitting. 
my.neuralnet.multilayer<-
  function(X1,Y,hidden=3,output="linear",num.layers=1,lambda=0)
  {
    X<-cbind(1,X1)
    input.layer.length<-length(X[1,])
    w01<-rnorm(input.layer.length*hidden+(num.layers-
                                            1)*hidden*hidden)
    w02<-rnorm(hidden)
    w0<-c(w01,w02)
    #print(w0)
    myoptfunc<-
      function(w0){my.eval2.nnet.ml(w0,X,Y,hidden,output,num.layers,lambda)$llik}
    dum<-optim(w0,myoptfunc,method="CG")
    wfinal<-dum$par
    ss<-dum$val
    duh<-
      my.eval2.nnet.ml(wfinal,X,Y,hidden,output,num.layers,lambda)
    plot(duh$pred,Y,
         main=paste("llik=",duh$llik,"\nSS=",ss),xlab="Prediction")
    list(ss=ss,wfinal=wfinal,ll=duh$llik)
  }
my.eval1.nnet.ml<-
  function(Xrow,w0,hidden,output,num.layers=1){
    input.layer.length<-length(Xrow)
    w01<-w0[c(1:(length(Xrow)*hidden))]
    w0A<-w0[-c(1:(length(Xrow)*hidden))]
    w1<-matrix(w01,input.layer.length,hidden)
    xhidden<-t(Xrow)%*%w1
    zhidden<-my.logistic(xhidden)
    if(num.layers==1){
      w02<-w0A
    }
    else{
      nlayers<-num.layers
      while(nlayers>1){
        w01<-w0A[c(1:(hidden*hidden))]
        w0A<-w0A[-c(1:(hidden*hidden))]
        w01<-matrix(w01,hidden,hidden)
        xhidden<-(zhidden)%*%w01
        zhidden<-my.logistic(xhidden)
        nlayers<-nlayers-1
      }
    }
    w02<-w0A
    out<-sum(w02*zhidden)
    if(output=="binary"){
      out<-my.logistic(out)
    }
    out
  }
my.eval2.nnet.ml<-
  function(w0,X,Y,hidden,output,num.layers,lambda)
  {
    zfunc<-
      function(V){my.eval1.nnet.ml(V,w0,hidden,output,
                                   num.layers)}
    pred<-apply(X,1,zfunc)
    if(output=="binary"){
      llik<-(-1)*sum(log(pred)*Y+log(1-pred)*(1-Y))
    }else{
      llik<-sum((pred-Y)^2)
    }
    loglik<-llik
    llik<-llik+lambda*sum(w0^2)
    list(llik=llik,pred=pred,y=Y,loglik=loglik)
  }
my.greedy.neuralnet<-
  function(X0,Y,hidden=3,output="linear",num.layers=1,lambda=0,trials=15)
{
  par(mfrow=c(4,4))
  dum1<-my.neuralnet.multilayer(X0,Y,hidden,output,num.layers,lambda)
  print(c(dum1$ss,dum1$ll))
  for(i in 1:(trials-1)){
    dum2<-my.neuralnet.multilayer(X0,Y,hidden,output,num.layers,lambda)
    print(c(dum2$ss,dum2$ll))
    if(dum2$ss<dum1$ss){
      dum1<-dum2
    }
  }
  dum0<-
    my.eval2.nnet.ml(dum1$wfinal,cbind(1,X0),Y,hidden,output,num.layers,lambda)
  plot(dum0$pred,Y,
       main=paste("llik=",dum0$llik,"\nSS=",dum1$ss),xlab="Prediction")
  dum1
}



#command from slides
#my.greedy.neuralnet(NOAA.newA[,c(-1)],NOAA.newA[,1],hidden=4,num.layers=2,lambda=0.003,63)
#Look at NOAAnewA, this is missing the last two years of NOAAnew
#Make NOAAnewB to match NOAAnewA but with those added two years and run the greedy neural net optimizer on it to fit the regression.

#new_row <- c(4.1, 20, 0.85, (4.1*4.1), (0.85*0.85),(4.1*0.85))
#NOAAnewB <- rbind(NOAAnewB, new_row)
#new_row <- c(4.2, 18, 0.89, (4.2*4.2), (0.89*0.89),(4.2*0.89))
#NOAAnewB <- rbind(NOAAnewB, new_row)
#View(NOAAnewB)
NOAAnewB<-as.matrix(NOAAnewB)
my.greedy.neuralnet(NOAAnewB[,c(-2)],NOAAnewB[,2],hidden=4,num.layers=2,lambda=0.003,63)
