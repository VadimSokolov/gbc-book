-- filters/chapter-shortref.lua
-- Replace chapter cross-reference display text "N Full Title" → "Ch N" (HTML only)
-- Sub-section refs like "5.3 Title" and bare numbers "3" are left unchanged.

function Link(el)
  if not quarto.doc.is_format("html") then return el end

  local is_xref = false
  for _, cls in ipairs(el.classes) do
    if cls == "quarto-xref" then is_xref = true; break end
  end
  if not is_xref then return el end

  local text = pandoc.utils.stringify(el.content)
  -- Match chapter-level refs: integer + space + non-whitespace (title follows)
  -- Does NOT match "5.3 Title" (has dot) or bare "3" (no trailing text)
  local num = text:match("^(%d+)%s+%S")
  if num then
    el.content = { pandoc.Str("Ch"), pandoc.Space(), pandoc.Str(num) }
    return el
  end

  return el
end
