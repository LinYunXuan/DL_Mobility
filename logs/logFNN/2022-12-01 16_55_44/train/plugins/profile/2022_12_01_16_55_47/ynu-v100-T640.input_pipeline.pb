	??8dI'@??8dI'@!??8dI'@	e?)z???e?)z???!e?)z???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??8dI'@?Fx@1Iط????AzUg????I???ދO@Y?gA(????*	O??n?h@2U
Iterator::Model::ParallelMapV2(???,??!?Ŵq?9@)(???,??1?Ŵq?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatH?9??*??!?p??_?5@)?O:?`???1w۰??F1@:Preprocessing2F
Iterator::Modelv3????!?O;@Y2D@)?c?ZB??1,??^?-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?z?΅??!B?y???8@)׿?3g}??1?V&??)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice F?6???!?ͨt(@) F?6???1?ͨt(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipwٯ;?y??!h?Ŀ??M@)ݱ?&???12)mM%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;???!?Uz?G?@);???1?Uz?G?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?{?O???!?Ve3Gd;@)???q??s?1!?\??u@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?45.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9e?)z???I]????V@Q? ?w?|@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Fx@?Fx@!?Fx@      ??!       "	Iط????Iط????!Iط????*      ??!       2	zUg????zUg????!zUg????:	???ދO@???ދO@!???ދO@B      ??!       J	?gA(?????gA(????!?gA(????R      ??!       Z	?gA(?????gA(????!?gA(????b      ??!       JGPUYe?)z???b q]????V@y? ?w?|@