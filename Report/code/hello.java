void main(String[] args)
{
  System.out.println("Hello World");
  List<String> a = new ArrayList<String>(); (*@ \label{line:jbug-start} @*)
  a.add("foo");
  List<Object> b = a; // This is a compile-time error (*@ \label{line:jbug-end} @*)
}
