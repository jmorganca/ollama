export type RawArtwork = {
  id: number;
  title: string;
  artist_display: string;
  place_of_origin: string;
  date_start: number;
  date_end: number;
  duration: number;
  dimensions: string;
  medium_display: string;
  credit_line: string;
  artwork_type_title: string;
  department_title: string;
  artist_title: string;
  classification_title: string;
  description: string;
}

export type Artwork = {
  id: number;
  title: string;
  country: string;
  date: number;
  artist: string;
  description: string;
}