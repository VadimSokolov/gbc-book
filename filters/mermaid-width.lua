local mermaid_width_pdf = 6.0   -- inches (letter 8.5in - 2x1in margins, slight inset)
local mermaid_width_html = "100%"

local function to_inches(s)
  if not s or s == "" then return nil end
  local n, unit = s:match("([%d%.]+)(%a*)")
  if not n then return nil end
  n = tonumber(n)
  if unit == "in" then return n end
  if unit == "pt" then return n / 72.27 end
  if unit == "cm" then return n / 2.54 end
  if unit == "mm" then return n / 25.4 end
  return nil
end

function CodeBlock(el)
  if el.classes:includes("mermaid") then
    if not el.attributes["fig-width"] then
      if quarto.doc.is_format("pdf") or quarto.doc.is_format("latex") then
        el.attributes["fig-width"] = tostring(mermaid_width_pdf)
      else
        el.attributes["fig-width"] = "8"
      end
    end
  end
  return el
end

function Image(el)
  if el.src:match("mermaid%-figure") then
    el.attributes["height"] = nil
    el.attributes["keepaspectratio"] = "true"
    el.attributes["fig-align"] = "center"

    if quarto.doc.is_format("pdf") or quarto.doc.is_format("latex") then
      local current_width = to_inches(el.attributes["width"])
      if not current_width or current_width > mermaid_width_pdf then
        el.attributes["width"] = tostring(mermaid_width_pdf) .. "in"
      end
    else
      if not el.attributes["width"] or el.attributes["width"] == "" then
        el.attributes["width"] = mermaid_width_html
      end
    end
  end
  return el
end
