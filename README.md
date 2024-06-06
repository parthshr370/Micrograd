# Micrograd
My implimentation of micrograd library from scratch that implements backpropagation 



1. Micrograd helps to perform Backpropagation on a NN
2. Backpropagation does is goes backwards in a feed forward NN and then applies chain rule to it 
3. NN Take Input Data as the input and the weights as an input and then perform mathematical operations 
4. Start by importing libraries 

```python 
import numpy as np
import matplotlib.pyplot as plt
import math
%matplotlib inline
```

5. Create a function ( probably a parabola)
```python 
def f(x):
return 3*x**2 - 4*x + 5
```

6. To plot the function with a certified range of ( -5,5) and finding when it reaches the minimum and then plot it on matplotlib

```python 
xs = np.arange(-5,5,0.25)
ys = f(xs)
ys

plt.plot(xs,ys)
```


7. We now need to calculate the derivative from scratch using simple basic math we know of 
8. We will see how x changes if we nudge the x with a small number h and then work see how it expands 
```python 
h = 0.001 
x = 3

f(x + h)
f(x+h) - f(x) 
```

9. The `f(x+h) - f(x)` tells you how much the function expanded with the `h`
10. To get the slope of the function we divide by `h`

```python 
h = 0.0000001
x = 2/3
(f(x+h) - f(x))/h

# 2.9753977059954195e-07
```

11. Finding Derivative for multiple numbers 
```python 
h = 0.00001

a = 10 
b = -4
c = 15

```
12. Starting a value object for micrograd 
```python 
class Value:
    def __init__(self,data):
        self.data = data
        
    def __repr__(self):
        return f"value is {self.data}"

a = Value(2.0)
b = Value(-3.0)

c = a + b # will not work as Value does not understand addition

```

13. We will introduce a new tuple class called children that will be a tuple 
14. Also a attribute named prev and op 
```python 
class Value:
    def __init__(self,data,_children=(),_op='',name=''):
        self.data= data
        self._prev = set(_children) # set is a datatype to store values 
        self._op = _op
        self.name = name 
        
    def __repr__(self):
        return f"value is {self.data}"
    
    
    def __add__(self,other):
        output = Value(self.data+ other.data, (self,other),'+','parth')
        return output
    
    def __mul__(self,other):
        out = Value(self.data*other.data, ( self,other),'*','rahul')
        return out
# a.__mul__(b) is what is going on behind the scenes 

```

15. Creating a new attribute named label that will label the elements that we perform addition or multiplication 
```python 
a = Value(2.0,label = 'a')
b = Value(-3.0, label = 'b' )
c = Value(10.0, label = 'c')
d = a*b + c ; d.label = 'd'
e = a*b; e.label = 'e'
f = Value(-2.0,label = 'f')
L= d*f ; L.label = 'L'
L
```

16. Use graphviz to start visualising the Data nodes and everything else as a Network
17.  We then try to manually perform backpropagation 
```python 
# create a function to test diff values of l1 l2 

def lol():
    
    h = 0.0001
    a = Value(2.0,label = 'a')
    b = Value(-3.0, label = 'b' )
    c = Value(10.0, label = 'c')
    d = a*b + c ; d.label = 'd'

    e = a*b; e.label = 'e'
    f = Value(-2.0,label = 'f')
    L= d*f ; L.label = 'L'
    L1 = L.data # creating L1
    
    a = Value(2.0+h,label = 'a')
    b = Value(-3.0, label = 'b' )
    c = Value(10.0, label = 'c')

    d = a*b + c ; d.label = 'd'

    e = a*b; e.label = 'e'
    f = Value(-2.0,label = 'f')
    L= d*f ; L.label = 'L'
    L2 = L.data # creating L2 with + h as a minor addition 
    
    
    print((L2-L1)/h) # this presents the derivative for small h
    

lol()

```

18. We need to revise some derivatives 
```
l = d * f
then dl/dd will be just f 
and dl/df will be just d 
so they are intechangebly present 
```

19. L is the final node and we need to find out the derrivative of each node ( except the input node ) wrt to L 
20. So we need to find out dL/dc now 
21. We can see how L interacts with all the previous nodes so we can find a pattern 
22. The derivative of a expression that uses the sum of two number is always 1
```python 
dd/dc = 1.0 
dd/de = 1.0
d = c + e 
```
23. Now we will introduce chain rule in it 
![Chain](https://mathsathome.com/wp-content/uploads/2021/10/the-chain-rule-in-words-1024x576.png)



24. Whatever we did till now was manual backpropagation 
- The recursive application of chain rule across the nodes of a network 
- We need L to go up in the gradient that is we will tweak each element a b c d e 
- Upto some small number 

25. Activation function is just a small nudge that guides the output out of a perceptron 
26. `Tanh` is an activation funciton 
```python
plt.plot(np.arange(-5,5,0.2) , np.tanh(np.arange(-5,5,0.2))) ; plt.grid
```

![tanh](https://www.medcalc.org/manual/functions/tanh.png)




27. To sort the graph we will use something called topological sort 
28. We will implement the topological sort in reverse order 
29. Now we will implement an actual backward function that will perform reversed topological sort on the NN
30. We run into a problem that if we use a variable more than once the gradient will just reset itself to 1 that will not be the actual gradient 
31. Solution to this is to accumulate these gradients ( ie perform addition )
32. 