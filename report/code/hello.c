int main(void)
{
  printf("Hello World\n");
  int a = *((volatile int *) 0); // uh-oh! (*@ \label{line:bug} @*)
  return 0;
}
