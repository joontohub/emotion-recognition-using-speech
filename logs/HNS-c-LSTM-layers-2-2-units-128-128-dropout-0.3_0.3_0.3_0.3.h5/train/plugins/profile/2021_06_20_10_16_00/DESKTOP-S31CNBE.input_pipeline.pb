  *	hffff?q@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Q???!Q?%? ?L@)Gr?鷿?1^MC???E@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapۊ?e????!2Tw???8@)f??a?֤?1$I?$I?,@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?<,Ԛ???!?_\|?$@)??q????1ޖQ?A?"@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenatej?t???!????/@)??A?f??1?ո?W@:Preprocessing2U
Iterator::Model::ParallelMapV2??ǘ????!?T???@)??ǘ????1?T???@:Preprocessing2F
Iterator::Model_?Qڛ?!ॊ?+#@)?I+???15??U?@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate???????!!3??RI@)?j+??݃?1p?~#?<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV-???!???8?[@)?j+??݃?1p?~#?<@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?*??	??!7ְP?zP@)?St$????1????qN@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch????Mb??!??-?v@)????Mb??1??-?v@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?s?!Ā??????)a2U0*?s?1Ā??????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP?s?b?!???h???)HP?s?b?1???h???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[1]::FromTensor??_?LU?!)v=?4??)??_?LU?1)v=?4??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[12]::Concatenate[1]::FromTensora2U0*?C?!Ā??????)a2U0*?C?1Ā??????:Preprocessing2?
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[13]::Concatenate[0]::TensorSlicea2U0*?C?!Ā??????)a2U0*?C?1Ā??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.