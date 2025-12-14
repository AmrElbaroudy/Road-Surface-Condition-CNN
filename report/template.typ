#let doc(title: none, authors: (), date: none, course: none, instructors: (), body) = {
  // Set the document's basic properties.
  set document(author: authors.flatten().join(", "), title: title)
  set text(font: "New Computer Modern", size: 12pt)
  set page(numbering: "1", number-align: center)

  // Title page
  v(1fr)
  text(2em, weight: 700, title)
  v(1em)
  text(1.2em, date)
  v(1em)
  text(1.2em, course)
  grid(
    columns: (1fr, 1fr),
    gutter: 1cm,
    ..instructors.map(instructor => text(1.2em, instructor))
  )

  v(1.2em)
  text(1.5em, "Authors:", weight: 700)
  grid(columns: (1fr, 1fr),
    gutter: 1cm,
    ..authors.flatten().map(author => text(1.2em, author))
  )
  v(1fr)



  // Main body.
  set par(justify: true)

  body
}
