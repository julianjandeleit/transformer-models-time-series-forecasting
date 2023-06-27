using Pkg
Pkg.activate(temp=true)
Pkg.add("Flux")
Pkg.add("Transformers")
Pkg.add("TensorBoardLogger")
Pkg.add("CUDA")
Pkg.add("Plots")
## 
begin
	using Flux
	using Transformers
	using TimeSeries
	using Flux
	using Flux: gradient
	using Flux.Optimise: update!
	using BSON: @save
	using TensorBoardLogger, Logging
	using CUDA
	using Statistics
	using Plots
	#using MarketData
end
## 
gpu_enabled = enable_gpu(true)
ta = readtimearray("transformers_jl/rate.csv", format="mm/dd/yy", delim=',')
## 
"""
Split sequence into feature and target and label parts
"""
function get_src_trg(
    sequence, 
    enc_seq_len, 
    dec_seq_len, 
    target_seq_len
)
	nseq = size(sequence)[2]
	
	@assert  nseq == enc_seq_len + target_seq_len
	src = sequence[:,1:enc_seq_len,:]
	trg = sequence[:,enc_seq_len:nseq-1,:]
	@assert size(trg)[2] == target_seq_len
	trg_y = sequence[:,nseq-target_seq_len+1:nseq,:]
	@assert size(trg_y)[2] == target_seq_len
	if size(trg_y)[1] == 1
	 	return src, trg, dropdims(trg_y; dims=1)
	else
		return src, trg, trg_y
	end
end

# begin
# 	## Model parameters
# 	dim_val = 512 # This can be any value. 512 is used in the original transformer paper.
# 	n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
# 	n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
# 	n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
# 	input_size = 1 # The number of input variables. 1 if univariate forecasting.
# 	dec_seq_len = 92 # length of input given to decoder. Can have any integer value.
# 	enc_seq_len = 153 # length of input given to encoder. Can have any integer value.
# 	output_sequence_length = 58 # Length of the target sequence, i.e. how many time steps should your forecast cover
# 	in_features_encoder_linear_layer = 2048 # As seen in Figure 1, each encoder layer has a feed forward layer. This variable determines the number of neurons in the linear layer inside the encoder layers
# 	in_features_decoder_linear_layer = 2048 # Same as above but for decoder
# 	max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder
	
# end

## Define Model

begin
	#define 2 layer of transformer
	input_size=16 # sequence length!
	decoder_input_size = 8 # seq len!
	num_layer = 8
	hidden_size = 32#128 # encoding of encoder
	num_head = 8
	head_hidden_size = div(hidden_size, num_head)
	intermediate_size = 4hidden_size
	output_sequence_length = 8
	# define two layer input
	encoderTransformerLayers = Transformer(Layers.TransformerBlock,
		num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size, return_score=true)|> gpu
	#define 2 layer of transformer decoder
	decoderTransformerLayers = Transformer(Layers.TransformerDecoderBlock,
	num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size;
	return_score = true) |> gpu
	
	
	encoder_input_layer = Dense(input_size,hidden_size) |> gpu
	decoder_input_layer = Dense(decoder_input_size,hidden_size) |> gpu
	positional_encoding_layer = Layers.SinCosPositionEmbed(hidden_size) |> gpu
	p = 0.2
	dropout_pos_enc = Dropout(p) |> gpu
	
	#define the layer to get the final output probabilities
	#linear = Positionwise(Dense(dim_val, output_sequence_length))
	linear = Dense(hidden_size,output_sequence_length) |> gpu
	function encoder_forward(x)
	  x = encoder_input_layer(x)
	  e = positional_encoding_layer(x)
	  t1 = x .+ e
	  t1 = dropout_pos_enc(t1)
	  t1 = encoderTransformerLayers((;hidden_state=t1, attention_mask=nothing))
	  return t1
	end
	
	function decoder_forward(tgt, t1)
	  decoder_output = decoder_input_layer(tgt)
	  t2 = decoderTransformerLayers((; hidden_state=decoder_output, memory=t1))
	  
	  #t2 = t2.hidden_state 
	  #t2 = Flux.flatten(t2.hidden_state)
	  #@show "linearize" t2=t2
	  p = linear(t2.hidden_state)
	  return p
	end
end
# #x = [1, 2, 3, 4, 5, 6, 7, 8] |> gpu
# y = [9, 10] |> gpu
# encoding = encoder_forward(x)

# prediction = decoder_forward(y, encoding)

## Data generation

"""
Generates sequences from data x
"""
function generate_seq(x, seq_len)
	result = Matrix{Float64}[]
	for i in 1:length(x)-seq_len+1
		ele = reshape(x[i:i+seq_len-1],(seq_len,1))	
		push!(result,ele)
	end
	return result
end

function loss(src, trg, trg_y)
	enc = encoder_forward(src)
	enc = enc.hidden_state #|> x->reshape(x,(128,32))
	dec = decoder_forward(trg, enc)
	err = Flux.Losses.mse(dec,trg_y)
	return err
end

begin
	ground_truth_curve = Vector{Float64}()
	for i in 1:500
		append!(ground_truth_curve,0.04*i+10)
		# append!(ground_truth_curve,sin(i))
	end
end

function normalize(data)
	dmin =minimum(data)
	dmax = maximum(data)
	normalizer(x) = (x-dmin)/(dmax-dmin)
	return normalizer.(data)
end
#generate_seq(sincurve1,enc_seq_len+output_sequence_length)
ta = readtimearray("data/preprocessed/timeseries.csv", format="yyyy-mm-dd HH:MM:SS", delim=',')
ta_mv = moving(mean,ta,15)
ground_truth_curve = [ta_mv...] .|> x->x[2][1]
ground_truth_curve = normalize(ground_truth_curve)

@show length(ground_truth_curve)


batch_size = 32
learning_rate = 1e-4
adam_betas = (0.9, 0.999)
lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)
# data = generate_seq(values(moving(mean,cl,15)),enc_seq_len+output_sequence_length)
# data = generate_seq(values(ta_mv[:"10 YR"]),enc_seq_len+output_sequence_length)
data = generate_seq(ground_truth_curve,input_size+output_sequence_length)
@show size(data)
data = reduce(hcat,data)
data = convert(Array{Float32,2}, data)
data_sz = size(data)
thd = floor(Int,data_sz[2]/2)
testdata = data[:,thd+1:end]
data = data[:,1:thd]

## Training
begin
	ps = Flux.params(encoder_input_layer, positional_encoding_layer, dropout_pos_enc, encoderTransformerLayers, decoder_input_layer, decoderTransformerLayers, linear)
	all_layers = [ encoder_input_layer, positional_encoding_layer, dropout_pos_enc, encoderTransformerLayers, decoder_input_layer, decoderTransformerLayers, linear ]
	opt = ADAM(learning_rate, adam_betas)
	train_loader = Flux.Data.DataLoader(data, batchsize=batch_size) 
	@info "start training"
	start_time = time()
	l = 100
	for i = 1:10 # num epochs (was 1000)
		for x in train_loader
			sz = size(x)
			sub_sequence = reshape(x,(1,sz[1],sz[2]))
		   	src, trg, trg_y = get_src_trg(
							    sub_sequence,
							    input_size,
							    decoder_input_size,
							    output_sequence_length
							    )
			src, trg, trg_y = todevice(src, trg, trg_y) #move to gpu
			
			# reshape because somehow model adds dimension at front
			src = src |> x->reshape(x,(input_size,size(src)[3])) |> gpu
			trg = trg |> x->reshape(x,(decoder_input_size,size(trg)[3])) |> gpu
			trg_y = trg_y |> gpu
			grad = gradient(()->loss(src, trg, trg_y), ps)
			Flux.update!(opt, ps, grad)
			global l = collect(loss(src, trg, trg_y))[1]
		    if l < 1e-3
				continue
			end
		end
		if i % 10 == 0		
			# for (j,layer) in enumerate(all_layers)
			# 	@save joinpath("tensorboard_logs","model-"*string(j)*".bson") layer
			# end
			with_logger(lg) do
				@info "train" loss=l log_step_increment=1
			
			end
			@info "train" i loss=l log_step_increment=1 (time() - start_time)
		end
		if l < 1e-3
				continue
		end
	end
end
## Test predictions

function predict(test_data)
    seq = Array{Float32}[]
    test_loader = Flux.Data.DataLoader(test_data, batchsize=batch_size)
    for x in test_loader
		sz = size(x)
        sub_sequence = reshape(x,(sz[1],sz[2]))
        ix = sub_sequence[1:input_size+output_sequence_length,:]
        ix = todevice(ix)
        enc = encoder_forward(ix[1:input_size,:])
        trg = ix[input_size:sz[1]-1,:]
        dec = decoder_forward(trg, enc)
        seq = vcat(seq,collect(dec[end,:]))
    end
    return seq
end



res = predict(testdata)
plot(res)
plot!(ground_truth_curve)


## Test specific data from above
x = [0.01, 0.02, 0.03, 0.04,0.05,0.06,0.07,0.08] |> gpu
y = [0.08, 0.09] |> gpu
encoding = encoder_forward(x)

prediction = decoder_forward(y, encoding) |> x->reshape(x,(2,1))

#loss(x, y, prediction |> x->reshape(x,(2,1)))
label = [0.09, 0.10] |> gpu
Flux.Losses.mse(label,prediction)