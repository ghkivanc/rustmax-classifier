<h1>Rustmax Classifier</h1>
is a pet project that I made because I wanted to implement some stuff I knew in a low level language.

<h2>Something like the Docs</h2>
<h3>1.The classifier itself</h3>
after importing the crate in. the classifier can be initialized by:
  SoftmaxClassifier::<num_features, num_classes>new();
<h3>2.preparing data</h3>
The 'library' does have some functions that make it possible to load any csv without headers to the dataformat classifier uses. load_real_data<num_features, num_classes>(absolutepath).
returns dataset,labels
<h3>3.training</h3>
call classifier.fit(&dataset, &labels, epochs, alpha)
<h3>4.inference</h3>
call classifier.infer(&data)

  
**Im aware this is not exactly a comperhensive documentation however as I stated this is just a pet project**
