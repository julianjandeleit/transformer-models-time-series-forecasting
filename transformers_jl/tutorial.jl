## 
using Flux
## 
actual(x) = 4x + 2
x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)

model = Dense(1 => 1)
model.weight, model.bias
## 
prediction = model(x_train)
using Statistics
loss(model, x, y) = mean(abs2.(model(x) .- y))
loss(model, x_train, y_train)
## 
using Flux: train!
opt = Descent()
data = [(x_train, y_train)]
train!(loss, model, data, opt)
loss(model, x_train, y_train)

for epoch in 1:200
    train!(loss, model, data, opt)
end

loss(model, x_train, y_train)

model(x_test)

y_test