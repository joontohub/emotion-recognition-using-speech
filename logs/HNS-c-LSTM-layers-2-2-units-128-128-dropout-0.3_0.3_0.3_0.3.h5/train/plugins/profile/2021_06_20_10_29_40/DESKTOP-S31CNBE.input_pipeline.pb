  *	23333o@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap7?[ A??!?܆u?*K@)9??v????1h???D@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapU???N@??!??c?sO>@)J+???1瑚?3@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???<,Ԛ?!1W?!?%@)J+???1瑚?#@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenatevq?-??!K???=y@)ŏ1w-!??1??Uo??@:Preprocessing2U
Iterator::Model::ParallelMapV29??v????!h???@)9??v????1h???@:Preprocessing2F
Iterator::Model
ףp=
??!A???I#"@)??~j?t??1?p+???@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?q?????!"?宱&	@)?q?????1"?宱&	@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate??H?}}?!4{6?h7@)?HP?x?1~c?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipU???N@??!??c?sON@)?~j?t?x?1V?r?X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata2U0*???!?N????@)?~j?t?x?1V?r?X@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??H?}m?!4{6?h7??)??H?}m?14{6?h7??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range_?Q?[?!?q8???)_?Q?[?1?q8???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensorǺ???F?!???????)Ǻ???F?1???????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensora2U0*?C?!?N??????)a2U0*?C?1?N??????:Preprocessing2?
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSlice-C??6:?!???????)-C??6:?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.