using Pkg
Pkg.activate(".")
Pkg.add("Flux")
Pkg.add("Transformers")
Pkg.add("TensorBoardLogger")
Pkg.add("CUDA")
Pkg.add("Plots")
Pkg.add("TimeSeries")
Pkg.add("BSON")
Pkg.add("StatsPlots")
Pkg.add("DataFrames")
# NOTE: potentially build cuda first
## 
begin
	using Flux
	using Transformers
	using TimeSeries
	using Flux
	using Flux: gradient
	using Flux.Optimise: update!
	using BSON: @save, @load
	using TensorBoardLogger, Logging
	using CUDA
	using Statistics
	using Plots
	using StatsPlots
	using DataFrames
	#using MarketData
end
## 
gpu_enabled = enable_gpu(true)
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

"""
with iterative predictions
"""
function get_src_trg_shifted(
    sequence, 
    enc_seq_len,
)
	src = sequence[1,1:enc_seq_len, :]
	trg = sequence[1,enc_seq_len+1:end, :]
	return src, trg
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
	num_layer = 3
	hidden_size = 128 # encoding of encoder
	num_head = 8
	head_hidden_size = div(hidden_size, num_head)
	intermediate_size = 4hidden_size
	output_sequence_length = 8
	# define two layer input
	encoderTransformerLayers = Transformer(Layers.TransformerBlock,
		num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size, return_score=true)|> todevice
	#define 2 layer of transformer decoder
	decoderTransformerLayers = Transformer(Layers.TransformerDecoderBlock,
	num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size;
	return_score = true) |> todevice
	
	
	encoder_input_layer = Dense(input_size,hidden_size) |> todevice
	decoder_input_layer = Dense(decoder_input_size,hidden_size) |> todevice
	positional_encoding_layer = Layers.SinCosPositionEmbed(hidden_size) |> todevice
	p = 0.2
	dropout_pos_enc = Dropout(p) |> todevice
	
	#define the layer to get the final output probabilities
	#linear = Positionwise(Dense(dim_val, output_sequence_length))
	linear = Dense(hidden_size,output_sequence_length) |> todevice
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
# #x = [1, 2, 3, 4, 5, 6, 7, 8] |> todevice
# y = [9, 10] |> todevice
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

function loss_shifted(src, trg)
	enc = encoder_forward(src)
	enc = enc.hidden_state
	#@show src, trg
	target_labels = vcat(src[end:end,:], trg[1:end-1,:]) # shift by one to left
	dec = decoder_forward(target_labels, enc)
	#@show dec |> size
	dec = reshape(dec, (size(dec)[1], size(dec)[2]))
	err = Flux.Losses.mse(dec,trg) # only trg, with unseen trg_y[end,:] gets predicted and evaluated
	return err
end

function loss_shifted_old(src, trg)
	enc = encoder_forward(src)
	enc = enc.hidden_state
	#@show src, trg
	target_labels = vcat(src[end:end,:], trg[1:end-1,:]) # shift by one to left
	dec = decoder_forward(target_labels, enc)
	#@show dec |> size
	dec = reshape(dec, (size(dec)[1], size(dec)[2]))
	err = Flux.Losses.mse(dec,trg) # only trg, with unseen trg_y[end,:] gets predicted and evaluated
	return err
end

begin
	ground_truth_curve = Vector{Float64}()
	for i in 1:500
		# append!(ground_truth_curve,0.04*i+10)
		append!(ground_truth_curve,sin(i*0.1))
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
ta_mv = moving(mean,ta,75)
ground_truth_curve = [ta_mv...] .|> x->x[2][1]
ground_truth_curve = normalize(ground_truth_curve)

@show length(ground_truth_curve)


batch_size = 2048
learning_rate = 1e-4#1e-4
adam_betas = (0.9, 0.999)
lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)
# data = generate_seq(values(moving(mean,cl,15)),enc_seq_len+output_sequence_length)
# data = generate_seq(values(ta_mv[:"10 YR"]),enc_seq_len+output_sequence_length)
#data = generate_seq(ground_truth_curve,input_size+n_pred_iteration*output_sequence_length)
data = generate_seq(ground_truth_curve,input_size+output_sequence_length)
@show size(data)
data = reduce(hcat,data)
data = convert(Array{Float32,2}, data)
data_sz = size(data)
thd = floor(Int,data_sz[2]/2)
testdata = data[:,thd+1:end]
data = data[:,1:thd]

## Training
best_layers = []
best_loss = 1e10
losses = []
begin
	ps = Flux.params(encoder_input_layer, positional_encoding_layer, dropout_pos_enc, encoderTransformerLayers, decoder_input_layer, decoderTransformerLayers, linear)
	all_layers = [ encoder_input_layer, positional_encoding_layer, dropout_pos_enc, encoderTransformerLayers, decoder_input_layer, decoderTransformerLayers, linear ]
	opt = ADAM(learning_rate, adam_betas)
	train_loader = Flux.Data.DataLoader(data, batchsize=batch_size) 
	@info "start training"
	start_time = time()
	l = 100
	for i = 1:100 # num epochs (was 100 for transpiration, 1000 for sin)
		for x in train_loader
			sz = size(x)
			sub_sequence = reshape(x,(1,sz[1],sz[2]))
		   	# src, trg, trg_y = get_src_trg(
			# 				    sub_sequence,
			# 				    input_size,
			# 				    decoder_input_size,
			# 				    n_pred_iteration*output_sequence_length
			# 				    )
			
			# src, trg, trg_y = todevice(src, trg, trg_y) #move to gpu
			# @show src, trg, trg_y
			# reshape because somehow model adds dimension at front
			src, trg = get_src_trg_shifted(sub_sequence, input_size) |> todevice
			#@show src, trg
			#src = src |> x->reshape(x,(input_size,size(src)[3])) |> todevice
			#trg = trg |> x->reshape(x,(decoder_input_size,size(trg)[3])) |> todevice
			#trg_y = trg_y |> todevice
			#@show src |> size, trg |> size
			#@show src, trg, trg_y
			grad = gradient(()->loss_shifted(src, trg), ps)
			Flux.update!(opt, ps, grad)
			global l = collect(loss_shifted(src, trg))[1]
			append!(losses, l)
			if l < best_loss
				#@info "best loss" l
				global best_loss = l
				global best_layers = all_layers |> deepcopy
			end
		    if l < 1e-3
				continue
			end
		end
		if i % 1 == 0		
			# for (j,layer) in enumerate(all_layers)
			# 	@save joinpath("tensorboard_logs","model-"*string(j)*".bson") layer
			# end
			with_logger(lg) do
				@info "train" loss=l log_step_increment=1
			
			end
			@info "train" i loss=l log_step_increment=1 (time() - start_time) best_loss learning_rate
		end
		if l < 1e-3
				continue
		end
	end
end


## Save data
encoder_input_layer, positional_encoding_layer, dropout_pos_enc, encoderTransformerLayers, decoder_input_layer, decoderTransformerLayers, linear = best_layers

for (j,layer) in enumerate(best_layers)
	_layer = layer |> cpu
	@save joinpath("data/experiment/layer$j.bson") _layer
end

@save joinpath("data/experiment/losses.bson") losses

## Load best model

loaded_layers = []
for i = 1:7
	@load "data/experiment/layer$i.bson" _layer
	push!(loaded_layers, _layer |> todevice)
end
@load "data/experiment/losses.bson" losses

encoder_input_layer, positional_encoding_layer, dropout_pos_enc, encoderTransformerLayers, decoder_input_layer, decoderTransformerLayers, linear = loaded_layers

## Test predictions



function predict(test_data)
    seq = Array{Float32}[]
    test_loader = Flux.Data.DataLoader(test_data, batchsize=batch_size)
	doshow=true
    for x in test_loader
		sz = size(x)
        sub_sequence = reshape(x,(sz[1],sz[2]))
        ix = sub_sequence[1:input_size+output_sequence_length,:]
        ix = todevice(ix)
        enc = encoder_forward(ix[1:input_size,:])
        trg = ix[input_size:sz[1]-1,:]
        dec = decoder_forward(trg, enc)
        seq = vcat(seq,collect(dec[end,:]))
		if doshow==true
			doshow=false
			#@show ix[1:input_size,:]
			ix=ix[1:input_size,1:1]
			trg=trg[:,1:1]
			dec=dec[:,1:1]
			#@show ix trg dec seq
		end
    end
    return seq
end

begin
	dataseq = data[1:24:end,:]'
	testdataseq = testdata[1:24:end,:]'
	res_train = predict(data)
	res_test = predict(testdata)
	#plot(ground_truth_curve;label="ground truth")
	pred_offset = input_size+decoder_input_size
	plot(pred_offset+1:pred_offset+length(res_train),res_train, label="train prediction", linewidth=2, dpi=300)
	plot!(pred_offset+length(dataseq)+1:pred_offset+length(dataseq)+length(testdataseq), res_test;label="test prediction",linewidth=2)
	plot!(dataseq;label="training data")
	plot!(length(dataseq)+1:length(dataseq)+length(testdataseq), testdataseq;label="test data")
	title!("predictions over ground truth")
	xlabel!("time")
	ylabel!("normalized transpiration") |> display
	savefig("data/experiment/predictions_over_gt.png")

	train_errors = dataseq .- res_train .|> abs |> x->[x...]
	@show train_errors |> median
	train_labels = ["train errors" for i = 1:length(train_errors)]
	test_errors = testdataseq .- res_test .|> abs |> x->[x...]
	@show test_errors |> median
	test_labels = ["test errors" for i = 1:length(train_errors)]
	boxplot([train_labels; test_labels], [train_errors; test_errors],outliers=false, label=false, dpi=300)
	# boxplot(errors, outliers=false, label="", whisker_width=:half, xaxis=false)
	ylabel!("absolute error")
	title!("prediction errors")
	savefig("data/experiment/prediction_errors.png")
end



## Predict iteratively
begin
	startindex = 1
	dataseq = collect(data[1:24:end,:]')[startindex:end]
	x = dataseq[1:input_size] |> todevice
	encoding = encoder_forward(x)
	target = [ dataseq[input_size+1:input_size+1decoder_input_size]...]
	rg = 20
	for i in 1:rg
		y =  target[end-decoder_input_size+1:end] |> todevice
		prediction = decoder_forward(y, encoding) # not working
		append!(target, prediction[end,1,1]) # dims 2,3 empty
	end
	# @show target
	target = target[decoder_input_size+1:end]


	dataseq = data[1:24:end,:]'
	testdataseq = testdata[1:24:end,:]'

	#plot(ground_truth_curve;label="ground truth")
	plot(startindex:startindex+length(x)-1,[x],linewidth=5)
	plot!(length(x)+startindex+decoder_input_size:decoder_input_size+startindex+length(x)+length(target)-1,target, label="target", linewidth=5)
	plot!(dataseq[1:1000])
	
	#plot!(dataseq;label="training data")
	#plot!(length(dataseq)+1:length(dataseq)+length(testdataseq), testdataseq;label="test data")
	title!("iterative prediction over ground truth")
	xlabel!("x")
	ylabel!("y")
end

## Predict iteratively with reencoding
begin
	startindex = 1
	dataseq = collect(data[1:24:end,:]')[startindex:end]
	x = dataseq[1:input_size] |> todevice
	encoding = encoder_forward(x)
	target = [ dataseq[input_size+1:input_size+1decoder_input_size]...]
	base = [x; target]
	@show target = [base...]
	rg = 20
	for i in 1:rg
		src = target[end-input_size-decoder_input_size+1:end-decoder_input_size]
		x = target[end-decoder_input_size+1:end]
		encoding = encoder_forward(src |> todevice).hidden_state
		#y = target[end-decoder_input_size+1:end] |> todevice
		prediction = decoder_forward(x |> todevice, encoding |> todevice) # not working

		append!(target, prediction[end,1,1]) # dims 2,3 empty
	end
	#@show target
	#target = target[length(base)+1:end] # remove base for visualization

	#plot(ground_truth_curve;label="ground truth")
	#plot(startindex:startindex+length(x)-1,[x],linewidth=5)
	#plot!(length(x)+startindex:startindex+length(x)+length(target)-1,target, label="target", linewidth=5)
	plot(dataseq[1:200])
	plot!(base)
	plot!(length(base)+1:length(base)+length(target),target)
	#plot!(dataseq;label="training data")
	#plot!(length(dataseq)+1:length(dataseq)+length(testdataseq), testdataseq;label="test data")
	title!("iterative prediction over ground truth (reencoded)")
	xlabel!("x")
	ylabel!("y")
end

## show losses
#plot(losses, thickness_scaling=2,size=(2500,2500))

plot(titlefontsize=40, tickfontsize=40, legendfontsize=40, guidefontsize=40)
plot!(losses, linewidth=2,size=(1000,1000))
xlabel!("test")
ylabel!("testy")
title!("title")
