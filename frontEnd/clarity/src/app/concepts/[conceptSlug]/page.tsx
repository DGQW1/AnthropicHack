// This is a server component
import ConceptClient from './ConceptClient';

export default function ConceptPage({ params }: { params: { conceptSlug: string } }) {
  // Directly access the conceptSlug from params (no await needed)
  const conceptSlug = params.conceptSlug;
  
  // Pass the extracted conceptSlug to the client component 
  return <ConceptClient conceptSlug={conceptSlug} />;
} 