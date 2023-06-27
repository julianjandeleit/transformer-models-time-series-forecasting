using Flux
using Transformers

num_layer = 1
hidden_size = 128
num_head = 4
head_hidden_size = div(hidden_size, num_head)
intermediate_size = 4hidden_size
# define two layer input
encoderA = Transformer(Layers.TransformerBlock,
    num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size, return_score=true)

decoder = Transformer(Layers.TransformerDecoderBlock,
num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size;
return_score = true)

function encoder_forward(x)
    x = encoder((;hidden_state=x))
end

## generate data
hidden_size
x = randn(Float32, 128, 10 #= sequence length =#, 1 #= batch size =#);
## apply

size(x)

encoder_forward(x)